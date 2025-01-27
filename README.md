# News Sentiment Analysis
The following is a demonstration of a Target-dependent Sentiment Classifier (TSC) on Maltese news articles. It uses a Hamborg's [BERT-based model](https://huggingface.co/fhamborg/roberta-targeted-sentiment-classification-newsarticles) trained on the [NewsMTSC](https://github.com/fhamborg/NewsMTSC) dataset.

This demonstration processes news articles from various Maltese newspapers, classifies the sentiment towards specific targets, and visualises the results.
The data is visualised in two ways:
1. **Aggregate Chart**: The `get_aggregate_sentiment` function generates an aggregate sentiment chart for a given target over time.
2. **Grouped by Newspaper**: The `get_grouped_sentiment` function generates sentiment charts grouped by newspaper, allowing for a comparison of sentiment across different sources.

The helper program sentiment/target_sentiment_classifier.py performs Named Entity Recognition (NER) using multiprocessing<sup>(*)</sup>, then performs TSC on the detected entities.


<sup>(*)</sup> Running multi-processing programs may not work on Google Colab.

---
## Funding
This work makes part of the NBxAI research project funded by [Xjenza Malta](https://xjenzamalta.mt/) via the Fusion R&I programme.
<p align="center">
    <img src="imgs/uom_ai.png" alt="Image 3" width="40%">
    <img src="imgs/fusion.png" alt="Image 1" width="25%">
    <img src="imgs/research_excellence.png" alt="Image 2" width="25%">
</p>


## Visualisation showcase
### Aggregate Sentiment Chart
![Robert Abela](imgs/1.png)
![Bernard Grech](imgs/2.png)
![Planning Authority](imgs/3.png)

### Sentiment Grouped by Newspaper
![Robert Abela by Newspaper](imgs/4.png)
