import base64
import json
from datetime import datetime
import requests
import os
import re
import urllib

from wasabi import msg

from goldenverba.components.document import Document
from goldenverba.components.interfaces import Reader
from goldenverba.server.ImportLogger import LoggerManager


class GitLabReader(Reader):
    """
    The GitLabReader downloads files from GitLab and ingests them into Weaviate.
    """

    def __init__(self):
        super().__init__()
        self.name = "GitLab"
        self.type = "URL"
        self.requires_env = ["GITLAB_TOKEN"]
        self.description = "Downloads only text files from a GitLab repository and ingests it into Verba. Use this format {owner}/{name}/{branch}/{folder} (e.g gitlab-org/gitlab/master/doc)"

    async def load(self, fileData: list, textValues: list[str], logger: LoggerManager) -> list[Document]:
        
        if len(textValues) <= 0:
            await logger.send_error(f"No GitLab Link detected")
            return []
        elif textValues[0] == "":
            await logger.send_error(f"Empty GitLab URL")
            return []

        gitlab_link = textValues[0]

        if not self.is_valid_gitlab_path(gitlab_link):
            await logger.send_error(f"GitLab URL {gitlab_link} not matching pattern: project_id/branch/folder")
            return []
        
        documents = []
        docs = await self.fetch_docs(gitlab_link, await logger)

        for _file in docs:
            try:
                await logger.send_info(f"Downloading {_file}")
                content, link, _path = self.download_file(gitlab_link, _file)
                if ".json" in _file:
                    json_obj = json.loads(str(content))
                    try:
                        document = Document.from_json(json_obj)
                    except Exception as e:
                        raise Exception(f"Loading JSON failed {e}")

                elif ".txt" in _file or ".md" in _file or ".mdx" in _file:
                    document = Document(
                        text=content,
                        type=self.config["document_type"].text,
                        name=_file,
                        link=link,
                        path=_path,
                        timestamp=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        reader=self.name,
                    )

                documents.append(document)

            except Exception as e:
                msg.warn(f"Couldn't load, skipping {_file}: {str(e)}")
                await logger.send_warning(f"Couldn't load, skipping {_file}: {str(e)}")
                continue

        return documents


    async def fetch_docs(self, path: str, logger: LoggerManager) -> list:
        split = path.split("/")
        project_path = urllib.parse.quote(split[0] + "/" + split[1], safe='')
        branch = split[2]
        folder_path = "/".join(split[3:]) if len(split) > 3 else ""

        url = f"https://gitlab.com/api/v4/projects/{project_path}/repository/tree?ref={branch}&path={folder_path}&per_page=100"
        headers = {
            "Authorization": f"Bearer {os.environ.get('GITLAB_TOKEN', '')}",
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        files = [
            item["path"]
            for item in response.json()
            if item["type"] == "blob"
            and (
                item["path"].endswith(".md")
                or item["path"].endswith(".mdx")
                or item["path"].endswith(".txt")
                or item["path"].endswith(".json")
            )
        ]

        msg.info(f"Fetched {len(files)} filenames from {url} (checking folder {folder_path})")

        logger.send_success(f"Fetched {len(files)} filenames from {url} (checking folder {folder_path})")

        
        return files

    def download_file(self, path: str, file_path: str) -> str:
        split = path.split("/")
        project_path = urllib.parse.quote(split[0] + "/" + split[1], safe='')
        branch = split[2]
        encoded_file_path = urllib.parse.quote(file_path, safe="")

        url = f"https://gitlab.com/api/v4/projects/{project_path}/repository/files/{encoded_file_path}/raw?ref={branch}"
        headers = {
            "Authorization": f"Bearer {os.environ.get('GITLAB_TOKEN', '')}",
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        content = response.text
        link = f"https://gitlab.com/{project_path}/-/blob/{branch}/{file_path}"
        msg.info(f"Downloaded {url}")
        return (content, link, file_path)

    def is_valid_gitlab_path(self, path):
        # Regex pattern to match project_id/branch/folder
        # {folder} is optional and can include subfolders
        pattern = r"^([^/]+)/([^/]+)(/[^/]+)*$"

        # Match the pattern with the provided path
        match = re.match(pattern, path)

        # Return True if the pattern matches, False otherwise
        return bool(match)