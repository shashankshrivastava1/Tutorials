import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile
    """

    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY ")}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


def get_data_from_gist():
    shashank_linkedin_gist = "https://gist.githubusercontent.com/shashan3/551096bd871bbf9a46c4a2dffe39b277/raw/9581e6fc1fc30ff402f8bfc9fb7b8c96591cbbd0/shashank_info_git.json"

    response = requests.get(shashank_linkedin_gist)
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


# https://www.linkedin.com/in/shashank-shrivastava-781b77b8/


# api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
# linkedin_profile_url = "https://www.linkedin.com/in/shashank-shrivastava-781b77b8/"
#
# api_key = '-2wMlBSLFNaBUizYKbCAPw'
# header_dic = {'Authorization': 'Bearer ' + api_key}
#
# response = requests.get(api_endpoint,
#                         params={'url': linkedin_profile_url},
#                         headers=header_dic)
