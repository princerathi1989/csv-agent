from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Load environment variables (make sure OPENAI_API_KEY is set in your .env)
load_dotenv()

def main():
    # Choose a model that works with agents

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4"  # works smoothly, avoids "stop" error
    )

    # Create the CSV agent
    csv_agent = create_csv_agent(
        llm=llm,
        path="episode_info.csv",   # replace with your CSV file path
        verbose=True,              # shows intermediate reasoning
        allow_dangerous_code=True  # lets agent run Python code for analysis
    )

    # Ask questions to the agent
    print("\n--- Query 1 ---")
    response1 = csv_agent.invoke(
        {"input": "How many columns are there in file episode_info.csv?"}
    )
    print(response1)

    print("\n--- Query 2 ---")
    response2 = csv_agent.invoke(
        {"input": "Print the seasons by ascending order of the number of episodes they have."}
    )
    print(response2)

if __name__ == "__main__":
    main()
