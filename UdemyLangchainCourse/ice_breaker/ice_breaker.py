from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# from agent.linkedin_lookup_agent import lookup as linked_in_lookup_agent
# from third_parties.linkedin import get_data_from_gist, scrape_linkedin_profile

information = """
Amitabh Bachchan (pronounced [əmɪˈt̪ɑːbʱ ˈbətːʃən]; born as Amita bh Shrivastav;[1] 11 October 1942[9]) is an Indian actor, film producer, television host, occasional playback singer and former politician, who works in Hindi cinema. In film career spanning over five decades, he has starred in more than 200 films. Bachchan is widely regarded as one of the most successful and influential actors in the history of Indian cinema.[10] Referred to as the Shahenshah of Bollywood, Sadi Ke Mahanayak (Hindi for, "Greatest actor of the century"), Star of the Millennium, or Big B.[11] His dominance in the Indian movie scene during the 1970s–1980s made the French director François Truffaut call it a "one-man industry".[12][relevant? – discuss]
"""


if __name__ == "__main__":
    print("hello Langchain")
    # data =get_data_from_gist()
    # data = scrape_linkedin_profile(
    #     "https://www.linkedin.com/in/shashank-shrivastava-781b77b8/"
    # )

    summary_template = """
         given the Linkedin information {information} about a person from I want you to create:
         1. a short summary
         2. two interesting facts about them
     """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # linkedin_profile_url = linked_in_lookup_agent(name="Shashank Shrivastava")
    #
    # linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    print(chain.run(information=information))