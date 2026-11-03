class AboutInfo:
    def __init__(self,
                 software_name: str,
                 version: str,
                 authors: list,
                 github_url: str = None,
                 documentation_url: str = None,
                 website_url: str = None,
                 license_info: str = None,
                 commit_info: str = None):
        """
        :param software_name: Name of the software
        :param version: Software version
        :param authors: List of author names
        :param github_url: URL to the GitHub repository
        :param documentation_url: URL to the documentation
        :param website_url: URL to the project website
        :param license_info: License text or short license description
        :param commit_info: Optional git commit info (hash, date and branch)
        """
        self.software_name = software_name
        self.version = version
        self.authors = authors
        self.github_url = github_url
        self.documentation_url = documentation_url
        self.website_url = website_url
        self.license_info = license_info
        self.commit_info = commit_info

    def to_html(self) -> str:
        """
        Returns a nicely formatted HTML string, with minimal styling so that
        it respects the current Qt theme (e.g., QDarkStyle).
        """
        # Convert list of authors to an HTML list
        authors_list = "".join(f"<li>{author}</li>" for author in self.authors)

        # Optional sections
        commit_id_section = (f"""
        <div class="commit-id">Commit ID: {self.commit_info}</div>
        """) if self.commit_info else ""

        github_section = (f"""
        <div class="section-title">GitHub Repository</div>
        <p>
          <a href="{self.github_url}" target="_blank">{self.github_url}</a>
        </p>
        """) if self.github_url else ""

        documentation_section = (f"""
        <div class="section-title">Documentation</div>
        <p>
          <a href="{self.documentation_url}" target="_blank">View Documentation</a>
        </p>
        """) if self.documentation_url else ""

        website_section = (f"""
        <div class="section-title">Website</div>
        <p>
          <a href="{self.website_url}" target="_blank">{self.website_url}</a>
        </p>
        """) if self.website_url else ""

        license_section = (f"""
        <div class="section-title">License</div>
        <p>
          {self.license_info}
        </p>
        """) if self.license_info else ""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8" />
          <style>
            body {{
              margin: 10px;
              /* No explicit color or background to respect theme */
            }}
            h1 {{
              margin-bottom: 0;
            }}
            .version {{
              font-size: 0.9em;
              /* color: use theme's default */
            }}
            .commit-id {{
              font-size: 0.85em;
              /* color: use theme's default */
            }}
            .section-title {{
              font-weight: bold;
              margin: 15px 0 5px;
            }}
            ul {{
              margin: 5px 0 15px 20px;
            }}
            a {{
              /* Let the style sheet or theme define link colors */
              text-decoration: none;
            }}
            a:hover {{
              text-decoration: underline;
            }}
            hr {{
              border: none;
              border-top: 1px solid;
              /* color: use theme's default */
              margin: 15px 0;
            }}
          </style>
        </head>
        <body>

          <h1>{self.software_name}</h1>
          <div class="version">Version: {self.version}</div>
          {commit_id_section}
          <hr />

          <div class="section-title">Authors</div>
          <ul>
            {authors_list}
          </ul>

          {github_section}
          {documentation_section}
          {website_section}
          {license_section}

        </body>
        </html>
        """

        return html_content.strip()
