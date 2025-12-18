# N8N Workflows

This folder contains N8N workflows used in this repository.

## DeepSeek Scraper

**File:** `Deepsek_Scrapper.json`

This workflow:
- Fetches essay links from Paul Graham's articles page  
- Downloads the essay text  
- Cleans and splits the content  
- Sends the text to DeepSeek for summarization  

## How to use

1. Start your N8N instance.
2. In the N8N UI, import `Deepsek_Scrapper.json` as a workflow.
3. Add your DeepSeek (or other model) credentials in N8N.
4. Run the workflow manually and inspect the output.

You can change the target URL, selectors, or model settings directly in the N8N editor.