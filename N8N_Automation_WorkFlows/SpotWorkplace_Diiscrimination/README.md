# Glassdoor Workplace Bias Analysis

![Workflow](https://raw.githubusercontent.com/Anujpatel04/Anuj-AI-ML-Lab/main/N8N_Automation_WorkFlows/SpotWorkplace_Diiscrimination/Workflow.png)

An automated n8n workflow that analyzes Glassdoor reviews to identify statistical disparities in workplace experiences across 19 demographic groups.

> **Note**: This project is part of the Anuj-AI-ML-Lab repository.

## Overview

Scrapes company reviews from Glassdoor, extracts demographic-specific ratings, performs statistical analysis (z-scores, effect sizes, p-values), and generates visual reports highlighting potential workplace bias patterns.

## Features

- **Automated Data Collection**: ScrapingBee bypasses JavaScript restrictions to extract Glassdoor data
- **Comprehensive Demographics**: Analyzes race, gender identity, sexual orientation, disability status, veteran status, and caregiving status (19 categories total)
- **Statistical Analysis**: Calculates z-scores, effect sizes, and p-values for each demographic group
- **AI Insights**: DeepSeek generates narrative analysis of significant disparities
- **Visual Reports**: Bar charts and scatter plots showing rating differences by group

## Prerequisites

- n8n (self-hosted or cloud instance)
- **ScrapingBee API** key ([Get API Key](https://www.scrapingbee.com)) - Web scraping with JS rendering
- **DeepSeek API** key ([Get API Key](https://platform.deepseek.com)) - AI data extraction & analysis
- **QuickChart** - No authentication required for data visualization

> **Note**: ScrapingBee free tier allows ~4-5 workflow runs per month (1,000 credits/month)

## Setup

1. Import the `SpotWorkplace_DiscriminationPatterns.json` workflow file into your n8n instance
2. Configure API credentials:
   - **ScrapingBee**: Add credentials for web scraping
   - **DeepSeek**: Add credentials for AI analysis
3. Update company name in the "SET company_name" node
4. Execute the workflow

## How It Works

1. **Search & Navigate**: Finds company on Glassdoor and navigates to reviews page
2. **Extract Data**: Scrapes overall ratings and demographic-specific ratings
3. **Calculate Statistics**: Computes variance, z-scores, effect sizes, and p-values
4. **Generate Reports**: Creates visualizations and AI-powered narrative analysis

## Statistical Metrics

- **Z-Score**: Standard deviations from mean (negative = worse experience)
- **Effect Size**: Magnitude of difference (>0.5 = large disparity)
- **P-Value**: Statistical significance (<0.05 = meaningful difference)

## Output

- Bar chart showing effect sizes by demographic
- Scatter plot of z-scores vs. effect sizes
- JSON data with complete statistical analysis
- AI-generated insights on workplace bias patterns

## Analyzed Demographics

- Race/Ethnicity
- Gender Identity
- Sexual Orientation
- Disability Status
- Caregiving Status
- Veteran Status

**Total**: 19 demographic categories

## Best Practices

- Use large US companies (500+ employees) for reliable data
- P-values <0.05 indicate statistically significant disparities
- Effect sizes >0.5 represent substantial differences
- Interpret results cautiously; statistics suggest patterns, not causation

## Limitations

- Limited by ScrapingBee free tier (1,000 credits/month)
- Requires companies with published demographic data
- Self-reported data may have selection bias
- Statistical significance requires adequate sample sizes

