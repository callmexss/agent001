import click

from agent.tools.summary_of_github import GithubRepoDataCollector


@click.command()
@click.argument("URL")
@click.argument("MODEL", default="gpt-3.5-turbo-0613")
@click.argument("TEMPERATURE", default=0.0)
def main(url: str, model: str, temperature: float):
    """Save github repo to json."""
    try:
        collector = GithubRepoDataCollector(url, model, temperature)
    except Exception as e:
        print(e)
        collector = GithubRepoDataCollector(url, "gpt-3.5-turbo-16k-0613", temperature)
    collector.print_and_save_repo_info()


if __name__ == "__main__":
    main()
