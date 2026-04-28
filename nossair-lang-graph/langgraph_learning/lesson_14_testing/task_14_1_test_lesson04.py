"""Task 14.1 — Test Your Lesson 4 Agent."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from lesson_04_tools_agent.lesson_04_tools_agent import agent_node, should_continue, tools, AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

class TestLesson04Agent(unittest.TestCase):
    def setUp(self):
        self.tools = tools
        self.tool_node = ToolNode(self.tools)
    
    def test_add_tool(self):
        """Test add tool returns correct sum."""
        result = self.tools[0].invoke({"a": 3, "b": 4})
        self.assertEqual(result, 7)
    
    def test_multiply_tool(self):
        """Test multiply tool returns correct product."""
        result = self.tools[1].invoke({"a": 5, "b": 6})
        self.assertEqual(result, 30)
    
    def test_weather_tool_known_city(self):
        """Test weather tool with known city."""
        result = self.tools[2].invoke({"city": "London"})
        self.assertIn("Cloudy", result)
    
    def test_weather_tool_unknown_city(self):
        """Test weather tool with unknown city."""
        result = self.tools[2].invoke({"city": "Mars"})
        self.assertIn("No weather data", result)
    
    def test_agent_node_runs(self):
        """Test agent node processes messages without error."""
        state = {
            "messages": [HumanMessage(content="What is 2 + 2?")],
            "tool_calls": []
        }
        result = agent_node(state)
        self.assertIn("messages", result)
        self.assertTrue(len(result["messages"]) > 0)
    
    def test_should_continue_with_tool_calls(self):
        """Test routing when tool calls present."""
        ai_msg = AIMessage(content="", tool_calls=[{"id": "1", "name": "add", "args": {"a": 1, "b": 2}}])
        state = {"messages": [ai_msg], "tool_calls": []}
        self.assertEqual(should_continue(state), "continue")
    
    def test_should_continue_without_tool_calls(self):
        """Test routing when no tool calls."""
        ai_msg = AIMessage(content="The answer is 4")
        state = {"messages": [ai_msg], "tool_calls": []}
        self.assertEqual(should_continue(state), "end")
    
    def test_graph_compiles(self):
        """Test graph compiles without errors."""
        from lesson_04_tools_agent.lesson_04_tools_agent import graph
        self.assertIsNotNone(graph)

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 14.1 — TESTING LESSON 4 AGENT")
    print("=" * 50)
    unittest.main(verbosity=2)
