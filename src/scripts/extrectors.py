from typing import List


def extract_posts(dataset: List[dict]) -> List[str]:
    """
    Extracts Facebook posts from a json formated data.

    Args:
        dataset (List[dict]): Json data.

    Returns:
          posts (List[str]): Facebook posts.
    """

    text = []

    for i in dataset:
        try:
            if ("har skrevet" in i["title"]) and ("tidslinje" in i["title"]):
                continue
        except (KeyError, TypeError, ValueError) as error:
            pass

        try:
            for j in i["data"]:
                for key in j.keys():
                    if key == "post":
                        text.append(j[key])
        except (KeyError, TypeError, ValueError) as error:
            pass

    return text


def extract_title(dataset: List[dict]) -> List[str]:
    """
    Extracts Google browse titles from a json formated data.

    Args:
        dataset (List[dict]): Json data.

    Returns:
          titles (List[str]): Google browse titles.
    """

    titles = []

    for i in dataset["Browser History"]:
        try:
            titles.append(i["title"])
        except (KeyError, TypeError, ValueError) as error:
            pass
    return titles
