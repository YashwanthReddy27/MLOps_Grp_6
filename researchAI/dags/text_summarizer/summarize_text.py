import requests
import yaml
class Summarize:
    def __init__(self):
        with open("/home/airflow/gcs/dags/common/config/secrets.yaml") as f:
            self.config = yaml.safe_load(f)
        self.api_token = self.config.get("HF_API")

    def summarize_news_descriptions(self, content, model_repo="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Summarizes long text using HF Router Inference API (chat completions).
        """

        API_URL = "https://router.huggingface.co/v1/chat/completions"
        api_token = self.api_token
        headers = {
            "Authorization": f"Bearer {api_token}",
        }

        prompt = f"""
        Understand the text and it's meaning. Don't add any other information apart from the given text in the summary.
        Summarize the following text into an expert human like breif summary:

        {content}
        """

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": model_repo,
            "max_tokens": 500,
            "temperature": 0.2
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        if summary:
            return summary.strip()
        return content


if __name__ == "__main__":
    summarizer = Summarize()
    long_text = (
        '''Artificial Intelligence (AI) has been making significant strides in recent years,
        transforming various industries and aspects of daily life. From healthcare to finance,
        AI technologies are being leveraged to improve efficiency, accuracy, and decision-making processes.
        Machine learning algorithms, natural language processing, and computer vision are some of the key
        areas where AI is being applied. As AI continues to evolve, it holds the potential to revolutionize
        the way we live and work, offering new opportunities and challenges alike.'''
    )
    summary = summarizer.summarize_news_descriptions(long_text)
    print("Summary:")
    print(summary)