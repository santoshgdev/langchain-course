from langchain_classic.agents import AgentExecutor
from libs.langchain.langchain_classic.agents.react.agent import create_react_agent

from doppler_map import doppler_map
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic_output_parser.prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

doppler_map()

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")
react_prompt = Client().pull_prompt("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tool_names", "agent_scratchpad", "input"]
).partial(format_instructions=output_parser.get_format_instructions())


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    result = agent_executor.invoke(
        {
            "input": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
                }
            ]
        }
    )
    # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()