import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()
client = OpenAI()

class Agent:
    """
    A simple ReAct (Reason + Act) Agent implementation.
    The agent follows a loop: Thought, Action, PAUSE, Observation.
    """
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages
        )
        return completion.choices[0].message.content

# --- Tools Implementation ---
def calculate(what):
    """Simple calculator tool using Python's eval."""
    return eval(what)

def average_dog_weight(name):
    """A mock database tool for dog weights."""
    if "scottish terrier" in name.lower():
        return "Scottish Terriers average weight is 20 lbs"
    elif "border collie" in name.lower():
        return "a Border Collies average weight is 37 lbs"
    elif "toy poodle" in name.lower():
        return "a toy poodles average weight is 7 lbs"
    else:
        return "An average dog weights 50 lbs"

# Registry of available tools
known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

# --- ReAct Prompt and Control Loop ---
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary.

average_dog_weight:
e.g. average_dog_weight: Collie
Returns average weight of a dog when given the breed.

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look up the dogs weight using average_dog_weight.
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs.

You then output:
Answer: A bulldog weights 51 lbs.
""".strip()

action_re = re.compile(r'^Action: (\w+): (.*)$')

def query(question, max_turns=5):
    """Executes the ReAct loop."""
    bot = Agent(prompt)
    next_prompt = question
    i = 0
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(f"--- Step {i} ---")
        print(result)
        
        # Look for actions in the assistant's response
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        
        if actions:
            # Action found, execute the tool
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            
            # Feed the observation back to the agent for the next turn
            next_prompt = f"Observation: {observation}"
        else:
            # No action found, assume the agent provided the final answer
            return

# --- Execution ---
if __name__ == "__main__":
    question = "I have 2 dogs, a border collie and a scottish terrier. What is their combined weight?"
    query(question)