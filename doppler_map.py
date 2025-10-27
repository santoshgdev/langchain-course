from os import environ


def doppler_map():
    environ['TAVILY_API_KEY'] = environ["SERVICE_TAVILY"]
    environ['OPENAI_API_KEY'] = environ['LLM_OPENAI']