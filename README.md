# [ANLP] Team Billie-Newman at SemEval-2023 Task 5: Clickbait Spoiling
This is the repository for the course 'Advanced Natural Language Processing' for the study 'Digital Sciences' at the University of Applied Sciences Cologne.
It contains the project code for the participation in the Clickbait Challenge proposed at SemEval-2023
  
``````
* Four Graph Implementations: Co-citation Graph, Citation Graph, Lotka-Inspired Graph, Journal Graph
* Metrics: **Degree Centrality**, **Closeness**, **Betweenness**, **Distance to most popular node**
* Comparing BM25 Baseline vs. Re-ranking with Boosting Factor
* Dataset: TREC-COVID / Cord19 dataset



#### Structure of this repository
* `data\`: Metadata used for the project
* `doc\`: Documentation and presentation of the project
* `scripts\`: Jupyter notebooks used for the project

#### Notebook description
| filename                    | description                                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Experiments.ipynb` | Creating, experimenting & evaluating retrieval perfomance with the graphs |
| `Scrape_metadata.ipynb` | Scraping relevant metadata using SemanticScholarAPI|


## Sources 
[cord19](https://ir-datasets.com/cord19.html)  
[Semantic Scholar](https://www.semanticscholar.org/)  
[pyterrier](https://github.com/terrier-org/pyterrier)  
