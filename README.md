# [ANLP] Team Billie-Newman at SemEval-2023 Task 5: Clickbait Spoiling
This is the repository for the course 'Advanced Natural Language Processing' for the study 'Digital Sciences' at the University of Applied Sciences Cologne.  
It contains the project code for the participation in the [Clickbait Challenge](https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html) proposed at SemEval-2023

* Task 1 Spoiler Classification: RoBERTa model with NER and custom components
* Task 2 Spoiler Generation: RoBERTa SQuAD2.0 model with rule-based approach
* Dataset: [Webis Clickbait Spoiling Corpus 2022](https://zenodo.org/record/6362726#.Y_np8B-ZNHV)
- - -
## Structure of this repository
* `doc\`: Contains the project presentation and project report
* `task1_anlp_deploy\`: Code and Docker File of Task 1
* `task2_anlp_deploy\`: Code and Docker File of Task 2

## File description
| filename                    | description                                                                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EDA.ipynb` | Code for pre-processing the WEBIS Clickbait Spoiling Corpus 2022 |
| `simple_transformer_task1.ipynb` | Code for training the RoBERTa model for multi-class classification |
| `run_task_1.py` | File for running the spoiler classifcation|
| `Reformat_to_SQuAD.ipynb` | Code for reformatting the spoiler questions into the SQuaD2.0 format |
| `Training_model.ipynb` | Code for training the RoBERTa-SQuAD2.0 model for the downstream task for spoiler generation|
| `run_task_2.py` | File for running the spoiler generation.  Arguments: --apply_rule_base v1 / --apply_rule_base v2|


## Docker Images
The docker images can be pulled from these dockerhub repositories:  
[[Task 1 Dockerhub Repo]](https://hub.docker.com/repository/docker/atran37/clickbait_task1_clf/general) | [[Task 2 Dockerhub Repo]](https://hub.docker.com/repository/docker/atran37/clickbait_task2_qa/general) 
#### Task 1 Command
```
docker run --rm -d >>>CONTAINER_NAME<<< --input >>INPUT_DATA<<<.jsonl --output output.jsonl --apply_ner=yes
```

#### Task 2 Command
Without rule-based approach:
```
docker run --rm -d >>>CONTAINER_NAME<<< --input >>INPUT_DATA<<<.jsonl --output output.jsonl --apply_rule_base=v1
```
With rule-based approach:
```
docker run --rm -d >>>CONTAINER_NAME<<< --input >>INPUT_DATA<<<.jsonl --output output.jsonl --apply_rule_base=v2
```
- - -
#### Sources 
[SemEval-2023 Task 5](https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html)  
[Webis Clickbait Spoiling Corpus 2022](https://zenodo.org/record/6362726#.Y_np8B-ZNHV) 
[pyterrier](https://github.com/terrier-org/pyterrier)  
