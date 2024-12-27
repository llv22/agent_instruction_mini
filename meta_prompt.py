import pandas as pd
import openai 
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel
from datasets import load_dataset
    
    
import matplotlib.pyplot as plt

client = openai.Client()

def get_model_response(messages, model="o1-preview"):
    response = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return response.choices[0].message.content

def generate_response(prompt): 
    messages = [{"role": "user", "content": prompt}]
    response = get_model_response(messages, model="gpt-4o-mini")
    return response

def generate_summaries(row):
    simple_itinerary = generate_response(simple_prompt.format(article=row["content"]))
    complex_itinerary = generate_response(complex_prompt + row["content"])
    return simple_itinerary, complex_itinerary

simple_prompt = "Summarize this news article: {article}"
meta_prompt = """
Improve the following prompt to generate a more detailed summary. 
Adhere to prompt engineering best practices. 
Make sure the structure is clear and intuitive and contains the type of news, tags and sentiment analysis.

{simple_prompt}

Only return the prompt.
"""

complex_prompt = get_model_response([{"role": "user", "content": meta_prompt.format(simple_prompt=simple_prompt)}])

evaluation_prompt = """
You are an expert editor tasked with evaluating the quality of a news article summary. Below is the original article and the summary to be evaluated:

**Original Article**:  
{original_article}

**Summary**:  
{summary}

Please evaluate the summary based on the following criteria, using a scale of 1 to 5 (1 being the lowest and 5 being the highest). Be critical in your evaluation and only give high scores for exceptional summaries:

1. **Categorization and Context**: Does the summary clearly identify the type or category of news (e.g., Politics, Technology, Sports) and provide appropriate context?  
2. **Keyword and Tag Extraction**: Does the summary include relevant keywords or tags that accurately capture the main topics and themes of the article?  
3. **Sentiment Analysis**: Does the summary accurately identify the overall sentiment of the article and provide a clear, well-supported explanation for this sentiment?  
4. **Clarity and Structure**: Is the summary clear, well-organized, and structured in a way that makes it easy to understand the main points?  
5. **Detail and Completeness**: Does the summary provide a detailed account that includes all necessary components (type of news, tags, sentiment) comprehensively?  


Provide your scores and justifications for each criterion, ensuring a rigorous and detailed evaluation.
"""

class ScoreCard(BaseModel):
    categorization: int
    keyword_extraction: int
    sentiment_analysis: int
    clarity_structure: int
    detail_completeness: int
    justification: str
    
def evaluate_summaries(row):
    simple_messages = [{"role": "user", "content": evaluation_prompt.format(original_article=row["content"], summary=row['simple_summary'])}]
    complex_messages = [{"role": "user", "content": evaluation_prompt.format(original_article=row["content"], summary=row['complex_summary'])}]
    
    simple_summary = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=simple_messages,
        response_format=ScoreCard)
    simple_summary = simple_summary.choices[0].message.parsed
    
    complex_summary = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=complex_messages,
        response_format=ScoreCard)
    complex_summary = complex_summary.choices[0].message.parsed
    
    return simple_summary, complex_summary

if __name__ == "__main__":
    ds = load_dataset("RealTimeData/bbc_news_alltime", "2024-08")
    df = pd.DataFrame(ds['train']).sample(n=100, random_state=1)
    print(complex_prompt)
    
    # Add new columns to the dataframe for storing itineraries
    df['simple_summary'] = None
    df['complex_summary'] = None

    # Use ThreadPoolExecutor to generate itineraries concurrently
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(generate_summaries, row): index for index, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Itineraries"):
            index = futures[future]
            simple_itinerary, complex_itinerary = future.result()
            df.at[index, 'simple_summary'] = simple_itinerary
            df.at[index, 'complex_summary'] = complex_itinerary

    df.head()

    # Add new columns to the dataframe for storing evaluations
    df['simple_evaluation'] = None
    df['complex_evaluation'] = None

    # Use ThreadPoolExecutor to evaluate itineraries concurrently
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate_summaries, row): index for index, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Summaries"):
            index = futures[future]
            simple_evaluation, complex_evaluation = future.result()
            df.at[index, 'simple_evaluation'] = simple_evaluation
            df.at[index, 'complex_evaluation'] = complex_evaluation

    df.head()

    df["simple_scores"] = df["simple_evaluation"].apply(lambda x: [score for key, score in x.model_dump().items() if key != 'justification'])
    df["complex_scores"] = df["complex_evaluation"].apply(lambda x: [score for key, score in x.model_dump().items() if key != 'justification'])


    # Calculate average scores for each criterion
    criteria = [
        'Categorisation',
        'Keywords and Tags',
        'Sentiment Analysis',
        'Clarity and Structure',
        'Detail and Completeness'
    ]

    # Calculate average scores for each criterion by model
    simple_avg_scores = df['simple_scores'].apply(pd.Series).mean()
    complex_avg_scores = df['complex_scores'].apply(pd.Series).mean()


    # Prepare data for plotting
    avg_scores_df = pd.DataFrame({
        'Criteria': criteria,
        'Original Prompt': simple_avg_scores,
        'Improved Prompt': complex_avg_scores
    })

    # Plotting
    ax = avg_scores_df.plot(x='Criteria', kind='bar', figsize=(6, 4))
    plt.ylabel('Average Score')
    plt.title('Comparison of Simple vs Complex Prompt Performance by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
