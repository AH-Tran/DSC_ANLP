# AGENDA
```

    TASK1 on Spoiler Type Classification: 
    The input is the clickbait post and the linked document. The task is to classify the spoiler type that the clickbait post warrants (either "phrase", "passage", "multi"). For each input, an output like {"uuid": "<UUID>", "spoilerType": "<SPOILER-TYPE>"} has to be generated where <SPOILER-TYPE> is either phrase, passage, or multi.

    TASK2 on Spoiler Generation: 
    The input is the clickbait post and the linked document (and, optional, the spoiler type if your approach uses this field). The task is to generate the spoiler for the clickbait post. For each input, an output like {"uuid": "<UUID>", "spoiler": "<SPOILER>"} has to be generated where <SPOILER> is the spoiler for the clickbait post.

```
AH 
- Figure out Docker Configuration with TIRA
- Adjust Code for TASK1

Kruff
- Data Exploration for Data Set
- Explore Methods for TASK2