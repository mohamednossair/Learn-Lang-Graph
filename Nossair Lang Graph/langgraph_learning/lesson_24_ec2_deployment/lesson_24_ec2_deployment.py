"""
Lesson 24: EC2 Production Deployment — Bringing It All Together
===============================================================
Teaches:
  - EC2-specific production deployment for the full architecture
  - IAM instance profile setup (replaces hardcoded credentials)
  - Nginx reverse proxy configuration for the FastAPI server
  - Systemd service for auto-restart on crash or EC2 reboot
  - Environment management with AWS SSM Parameter Store (secure config)
  - Health check endpoints for EC2 Auto Scaling / Load Balancer target groups
  - Zero-downtime deployment with rolling restart
  - CloudWatch logging integration (structured JSON logs → CloudWatch)
  - Combining all lessons: Bedrock (L21) + S3 (L22) + Conv API (L23) on EC2
  - Production startup checklist

Architecture role (from High Level Architecture diagram):
    Amazon ML – Greenfield
        └── EC2 Instance – FastAPI Server  ← THIS LESSON
               └── Nginx → uvicorn → FastAPI (L23)
                      ├── Agent Orchestration → Bedrock (L21)
                      ├── S3 storage (L22)
                      └── Oracle 19c / Memory (L16)

Prerequisites:
    pip install boto3 watchtower  (watchtower = CloudWatch log handler)

Environment variables managed via SSM Parameter Store (see SECTION 3):
    /p5/prod/JWT_SECRET_KEY
    /p5/prod/S3_BUCKET_NAME
    /p5/prod/BEDROCK_MODEL_ID
    /p5/prod/ORACLE_DSN
"""

import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from typing import Any, Optional

# ===========================================================================
# SECTION 1 — PRODUCTION LOGGING SETUP
# ===========================================================================
#
# In EC2 production, logs flow:
#   uvicorn → Python logging → CloudWatch Logs
#
# The CloudWatch handler (watchtower) is a drop-in Python logging handler.
# JSON structured format (from Lesson 18) is preserved — CloudWatch Insights
# can query JSON fields directly.

class CloudWatchJsonFormatter(logging.Formatter):
    """
    JSON log formatter that adds EC2 instance metadata to every log line.
    CloudWatch Logs Insights can then filter by instance_id, service, level, etc.
    """

    def __init__(self, service_name: str = "p5-chatbot-api"):
        super().__init__()
        self.service_name = service_name
        self.hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "host": self.hostname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, "tenant_id"):
            log_entry["tenant_id"] = record.tenant_id
        return json.dumps(log_entry)


def setup_production_logging(service_name: str = "p5-chatbot-api", log_group: str = "/p5/chatbot") -> logging.Logger:
    """
    Configure logging for EC2 production:
      - Console handler: JSON format (captured by systemd → journalctl)
      - CloudWatch handler: sends to /p5/chatbot log group (if boto3 available)
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CloudWatchJsonFormatter(service_name))
    logger.addHandler(console_handler)

    try:
        import boto3
        import watchtower
        cw_handler = watchtower.CloudWatchLogHandler(
            log_group=log_group,
            stream_name=f"{socket.gethostname()}-{datetime.utcnow().strftime('%Y%m%d')}",
            boto3_client=boto3.client("logs", region_name=os.getenv("AWS_REGION", "us-east-1")),
        )
        cw_handler.setFormatter(CloudWatchJsonFormatter(service_name))
        logger.addHandler(cw_handler)
        logger.info("CloudWatch logging enabled: %s", log_group)
    except ImportError:
        logger.info("watchtower not installed — CloudWatch logging disabled")
    except Exception as e:
        logger.warning("CloudWatch log handler init failed: %s — console-only logging", e)

    return logger


logger = setup_production_logging()


# ===========================================================================
# SECTION 2 — IAM INSTANCE PROFILE (CREDENTIAL STRATEGY)
# ===========================================================================
#
# On EC2, NEVER use access keys. Instead:
#
#   1. Create an IAM role, e.g. "ChatbotEC2Role"
#   2. Attach these policies:
#      - AmazonBedrockFullAccess  (or a scoped custom policy)
#      - AmazonS3FullAccess       (or scoped to your bucket ARN)
#      - AmazonSSMReadOnlyAccess  (for SSM Parameter Store config)
#      - CloudWatchLogsFullAccess (for log shipping)
#   3. Attach the role to your EC2 instance (Instance Settings → IAM Role)
#   4. boto3 auto-discovers the role via instance metadata (169.254.169.254)
#
# boto3 credential resolution order (same chain as L21, L22):
#   env vars → ~/.aws/credentials → instance metadata → container metadata
#
# LOCAL DEV: use aws configure or AWS_PROFILE env var
# EC2 PROD:  role is auto-injected — zero credentials in code or env

def verify_iam_role_available() -> bool:
    """
    Check if running on EC2 with an IAM role attached.
    Returns True if instance metadata is accessible (EC2 with role).
    Returns False in local dev (falls back to profile/env credentials).
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/iam/info",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            info = json.loads(resp.read())
            role_name = info.get("InstanceProfileArn", "").split("/")[-1]
            logger.info("Running on EC2 with IAM role: %s", role_name)
            return True
    except Exception:
        logger.info("Not on EC2 (or no IAM role) — using local credentials")
        return False


# ===========================================================================
# SECTION 3 — SSM PARAMETER STORE (SECURE CONFIG MANAGEMENT)
# ===========================================================================
#
# AWS Systems Manager Parameter Store is the production alternative to .env files.
#
# Why SSM instead of .env on EC2?
#   - Parameters are encrypted (SecureString type uses KMS)
#   - Audit trail: CloudTrail logs every GetParameter call
#   - No secrets in AMI snapshots or launch templates
#   - Centralized: all EC2 instances in the same region share the same params
#   - Access controlled by IAM (your EC2 role only reads its own params)
#
# Parameter naming convention: /{app}/{env}/{key}
#   /p5/prod/JWT_SECRET_KEY
#   /p5/prod/S3_BUCKET_NAME
#   /p5/dev/JWT_SECRET_KEY

SSM_PARAMETER_PREFIX = os.getenv("SSM_PARAMETER_PREFIX", "/p5/prod")


def load_config_from_ssm(keys: list[str]) -> dict[str, str]:
    """
    Load multiple configuration values from SSM Parameter Store.
    Falls back to environment variables if SSM is unavailable (local dev).

    Args:
        keys: List of parameter names (without prefix), e.g. ["JWT_SECRET_KEY", "S3_BUCKET_NAME"]

    Returns:
        Dict of key → value.
    """
    config = {}
    try:
        import boto3
        ssm = boto3.client("ssm", region_name=os.getenv("AWS_REGION", "us-east-1"))

        full_names = [f"{SSM_PARAMETER_PREFIX}/{k}" for k in keys]
        response = ssm.get_parameters(Names=full_names, WithDecryption=True)

        for param in response.get("Parameters", []):
            short_key = param["Name"].split("/")[-1]
            config[short_key] = param["Value"]
            logger.info("Loaded SSM param: %s", param["Name"])

        for param in response.get("InvalidParameters", []):
            short_key = param.split("/")[-1]
            logger.warning("SSM param not found, using env: %s", param)
            config[short_key] = os.getenv(short_key, "")

    except Exception as e:
        logger.info("SSM unavailable (%s) — loading all config from environment", e)
        config = {k: os.getenv(k, "") for k in keys}

    return config


def load_production_config() -> dict[str, str]:
    """
    Load the full production configuration for the Chatbot API.
    Merges SSM values with environment variable fallbacks.
    """
    return load_config_from_ssm([
        "JWT_SECRET_KEY",
        "S3_BUCKET_NAME",
        "BEDROCK_MODEL_ID",
        "ORACLE_DSN",
        "AWS_REGION",
    ])


# ===========================================================================
# SECTION 4 — HEALTH CHECK & READINESS
# ===========================================================================
#
# EC2 Auto Scaling and Application Load Balancer (ALB) require health checks.
# The target group pings GET /health every N seconds.
# If it returns non-200 twice in a row, EC2 is marked unhealthy and replaced.
#
# Two types:
#   /health/live   — is the process running? (liveness — only fails if process is hung)
#   /health/ready  — are all dependencies reachable? (readiness — fails if DB/S3 down)

_startup_time = time.time()


def check_bedrock_connectivity() -> dict:
    """Check if Bedrock is reachable from this EC2 instance."""
    try:
        import boto3
        client = boto3.client("bedrock", region_name=os.getenv("AWS_REGION", "us-east-1"))
        client.list_foundation_models(byOutputModality="TEXT")
        return {"status": "ok", "latency_ms": 0}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_s3_connectivity() -> dict:
    """Check if S3 bucket is accessible from this EC2 instance."""
    try:
        import boto3
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        bucket = os.getenv("S3_BUCKET_NAME", "my-langgraph-bucket")
        s3.head_bucket(Bucket=bucket)
        return {"status": "ok", "bucket": bucket}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_liveness() -> dict:
    """
    Liveness check: is the process alive and responding?
    Should NEVER check external dependencies — only internal process state.
    """
    return {
        "status": "alive",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


def get_readiness() -> dict:
    """
    Readiness check: are all external dependencies reachable?
    ALB removes the instance from rotation if this returns unhealthy.
    """
    bedrock = check_bedrock_connectivity()
    s3 = check_s3_connectivity()

    all_healthy = bedrock["status"] == "ok" and s3["status"] == "ok"

    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": {
            "bedrock": bedrock,
            "s3": s3,
        },
        "hostname": socket.gethostname(),
        "checked_at": datetime.utcnow().isoformat(),
    }


# ===========================================================================
# SECTION 5 — NGINX CONFIGURATION TEMPLATE
# ===========================================================================
#
# Nginx sits in front of uvicorn on EC2:
#   Internet → ALB → Nginx (port 80/443) → uvicorn (port 8000)
#
# Benefits:
#   - SSL termination at Nginx (HTTPS)
#   - Request buffering (protects uvicorn from slow clients)
#   - Rate limiting at Nginx level (before Python code)
#   - Static file serving (if any)
#   - Zero-downtime reload: nginx -s reload

NGINX_CONFIG_TEMPLATE = """
# /etc/nginx/sites-available/p5-chatbot
# Generated by lesson_24_ec2_deployment.py

upstream p5_api {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name _;

    # Security headers
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Nginx-level rate limiting (before reaching Python)
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/m;
    limit_req zone=api_limit burst=10 nodelay;

    location / {
        proxy_pass http://p5_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # SSE streaming — disable proxy buffering
    location /chat/stream {
        proxy_pass http://p5_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # Health check bypass — ALB pings this directly
    location /health {
        proxy_pass http://p5_api;
        access_log off;
    }
}
"""


def generate_nginx_config(output_path: str = None) -> str:
    """
    Generate the Nginx config and optionally write it to a file.
    On EC2: sudo python lesson_24.py --nginx > /etc/nginx/sites-available/p5-chatbot
    """
    if output_path:
        with open(output_path, "w") as f:
            f.write(NGINX_CONFIG_TEMPLATE)
        logger.info("Nginx config written to: %s", output_path)
    return NGINX_CONFIG_TEMPLATE


# ===========================================================================
# SECTION 6 — SYSTEMD SERVICE TEMPLATE
# ===========================================================================
#
# Systemd ensures uvicorn restarts if it crashes, and starts on EC2 boot.
# Key settings:
#   Restart=always          — restart on any exit (crash or OOM)
#   RestartSec=5            — wait 5s before restart (avoid tight restart loops)
#   EnvironmentFile         — loads /etc/p5/.env (non-secret config)
#   WorkingDirectory        — must match where the app code lives

SYSTEMD_SERVICE_TEMPLATE = """
# /etc/systemd/system/p5-chatbot.service
# Generated by lesson_24_ec2_deployment.py

[Unit]
Description=Chatbot API — LangGraph Multi-Agent FastAPI Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/p5-chatbot
EnvironmentFile=/etc/p5/.env
ExecStart=/opt/p5-chatbot/.venv/bin/uvicorn \\
    lesson_23_conversation_api.lesson_23_conversation_api:app \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --workers 4 \\
    --log-config /opt/p5-chatbot/log_config.json
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=p5-chatbot
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
"""


def generate_systemd_service(output_path: str = None) -> str:
    """Generate the systemd unit file for the Chatbot service."""
    if output_path:
        with open(output_path, "w") as f:
            f.write(SYSTEMD_SERVICE_TEMPLATE)
        logger.info("Systemd service written to: %s", output_path)
    return SYSTEMD_SERVICE_TEMPLATE


# ===========================================================================
# SECTION 7 — ZERO-DOWNTIME DEPLOYMENT PROCEDURE
# ===========================================================================
#
# Zero-downtime rolling restart on EC2:
#
#   OPTION A: Single EC2 (graceful reload)
#   ----------------------------------------
#   1. git pull (or aws s3 cp new code from S3)
#   2. pip install -r requirements.txt --quiet
#   3. sudo systemctl reload p5-chatbot     ← sends SIGHUP, uvicorn reloads workers
#      OR:
#      sudo systemctl restart p5-chatbot    ← brief downtime (< 1s with Nginx buffering)
#
#   OPTION B: Multi-EC2 behind ALB (true zero-downtime)
#   -----------------------------------------------------
#   1. Create new EC2 from updated AMI (or use launch template)
#   2. Register new instance in ALB target group
#   3. Wait for health checks to pass (/health/ready returns 200)
#   4. Deregister old EC2 from ALB (existing connections drain by default 300s)
#   5. Terminate old EC2
#
#   OPTION C: AWS CodeDeploy (automated)
#   --------------------------------------
#   - appspec.yml defines lifecycle hooks
#   - BeforeInstall: stop old service, install deps
#   - ApplicationStart: start new service
#   - ValidateService: curl /health/ready
#   - CodeDeploy handles ALB registration/deregistration automatically

DEPLOYMENT_CHECKLIST = """
EC2 PRODUCTION DEPLOYMENT CHECKLIST
=====================================

PRE-DEPLOYMENT:
  [ ] IAM role attached to EC2 instance (ChatbotEC2Role)
  [ ] SSM parameters populated: JWT_SECRET_KEY, S3_BUCKET_NAME, BEDROCK_MODEL_ID
  [ ] S3 bucket created with correct IAM permissions
  [ ] Bedrock model access enabled in AWS Console
  [ ] Oracle/DB connection tested from EC2 subnet
  [ ] Security group: port 80/443 open inbound, port 8000 NOT open to internet

INSTALLATION:
  [ ] Python 3.11+ installed
  [ ] pip install -r requirements.txt (in virtualenv)
  [ ] Nginx installed and configured (/etc/nginx/sites-available/p5-chatbot)
  [ ] Nginx config tested: sudo nginx -t
  [ ] Systemd service installed: /etc/systemd/system/p5-chatbot.service
  [ ] sudo systemctl daemon-reload
  [ ] sudo systemctl enable p5-chatbot  # start on boot
  [ ] sudo systemctl start p5-chatbot

VERIFICATION:
  [ ] curl http://localhost:8000/health/live   → {"status": "alive"}
  [ ] curl http://localhost:8000/health/ready  → {"status": "ready"}
  [ ] curl http://localhost/health             → via Nginx
  [ ] ALB target group shows instance as "healthy"
  [ ] CloudWatch Logs: /p5/chatbot log group receiving events
  [ ] Test session creation: POST /sessions
  [ ] Test chat: POST /chat

MONITORING:
  [ ] CloudWatch Alarm: CPU > 80% for 5 minutes
  [ ] CloudWatch Alarm: /health/ready returns non-200 (via Route53 health check)
  [ ] CloudWatch Dashboard: request count, latency p99, error rate
  [ ] SNS notification on alarms → team Slack/email
"""


# ===========================================================================
# SECTION 8 — FULL PRODUCTION STARTUP SEQUENCE
# ===========================================================================

def production_startup() -> bool:
    """
    Run all pre-flight checks before the FastAPI server starts.
    Called from uvicorn startup event (add to Lesson 23 FastAPI app lifespan).

    Returns True if all checks pass, False if any critical check fails.
    """
    logger.info("=== Chatbot API — Production Startup ===")

    on_ec2 = verify_iam_role_available()
    logger.info("EC2 IAM role: %s", "YES" if on_ec2 else "NO (local dev mode)")

    config = load_production_config()
    logger.info("Config loaded: %d parameters", len([v for v in config.values() if v]))

    readiness = get_readiness()
    for service, check in readiness["checks"].items():
        status = check["status"]
        logger.info("Dependency check [%s]: %s", service, status)

    critical_ok = all(
        check["status"] == "ok"
        for check in readiness["checks"].values()
    )

    if critical_ok:
        logger.info("All startup checks passed. Server ready.")
    else:
        logger.warning("Some dependency checks failed — starting in degraded mode.")

    return critical_ok


# ===========================================================================
# SECTION 9 — ENVIRONMENT FILE FOR NON-SECRET CONFIG
# ===========================================================================
#
# /etc/p5/.env (loaded by systemd EnvironmentFile directive):
# Contains NON-SECRET environment variables.
# Secrets (JWT_SECRET_KEY, DB password) come from SSM Parameter Store.
#
# This file is safe to include in the AMI or user data script.

ENV_FILE_TEMPLATE = """
# /etc/p5/.env
# Non-secret configuration for Chatbot API
# Secrets are loaded from SSM Parameter Store at runtime

AWS_REGION=us-east-1
SSM_PARAMETER_PREFIX=/p5/prod
MAX_MESSAGES_PER_SESSION=100
MAX_SESSIONS_PER_TENANT=50
RATE_LIMIT_RPM=30
S3_CONVERSATION_PREFIX=conversations/
S3_DOCUMENTS_PREFIX=documents/
LOG_LEVEL=INFO
WORKERS=4
"""


# ===========================================================================
# SECTION 10 — DEMO
# ===========================================================================

def run_demo():
    print("\n" + "=" * 65)
    print("LESSON 24 — EC2 PRODUCTION DEPLOYMENT DEMO")
    print("=" * 65)

    print("\n--- Check 1: IAM Role Detection ---")
    on_ec2 = verify_iam_role_available()
    print(f"Running on EC2 with IAM role: {on_ec2}")

    print("\n--- Check 2: Configuration Loading ---")
    config = load_production_config()
    for k, v in config.items():
        display = v[:20] + "..." if len(v) > 20 else v if v else "[not set]"
        print(f"  {k}: {display}")

    print("\n--- Check 3: Liveness ---")
    live = get_liveness()
    print(f"  Status: {live['status']}")
    print(f"  Uptime: {live['uptime_seconds']}s")
    print(f"  Host: {live['hostname']} (PID {live['pid']})")

    print("\n--- Check 4: Readiness (dependency connectivity) ---")
    ready = get_readiness()
    print(f"  Overall: {ready['status']}")
    for svc, check in ready["checks"].items():
        print(f"  [{svc}] {check['status']}")

    print("\n--- Check 5: Nginx Config Preview ---")
    nginx = generate_nginx_config()
    print(nginx[:300] + "...")

    print("\n--- Check 6: Systemd Service Preview ---")
    svc = generate_systemd_service()
    print(svc[:300] + "...")

    print("\n--- Deployment Checklist ---")
    print(DEPLOYMENT_CHECKLIST)

    print("\n" + "=" * 65)
    print("KEY TAKEAWAYS:")
    print("  1. IAM role on EC2 = zero credentials in code or environment files")
    print("  2. SSM Parameter Store = encrypted secrets, auditable, centralized")
    print("  3. /health/live vs /health/ready: different purposes, both required for ALB")
    print("  4. Nginx in front of uvicorn: SSL, rate limit, buffering, SSE support")
    print("  5. Systemd = auto-restart on crash + start on EC2 boot")
    print("  6. CloudWatch + watchtower: JSON logs flow from Python → CloudWatch")
    print("  7. Zero-downtime: single EC2 = systemctl reload; multi-EC2 = ALB drain + swap")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
    print("\nDeployment files generated:")
    print("  Nginx config  → generate_nginx_config('/etc/nginx/sites-available/p5-chatbot')")
    print("  Systemd svc   → generate_systemd_service('/etc/systemd/system/p5-chatbot.service')")
    print("  Env file      → /etc/p5/.env  (copy ENV_FILE_TEMPLATE)")
