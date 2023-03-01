# MultiCoNER
>Complex named entities (NE), like the titles of creative works, are not simple nouns and pose challenges for NER systems (Ashwini and Choi, 2014). They can take the form of any linguistic constituent, like an imperative clause (“Dial M for Murder”), and do not look like traditional NEs (Persons, Locations, etc.).

This repository contains solution for *[SemEval 2023 Task 2: MultiCoNER II
Multilingual Complex Named Entity Recognition](https://multiconer.github.io/)* and **will contain additional research of Multilingual Named Entity Recognition approaches**.

## Dataset


The tagset of MultiCoNER is a fine-grained tagset.

The fine to coarse level mapping of the tags are as follows:

    **Location (LOC) : Facility, OtherLOC, HumanSettlement, Station
    Creative Work (CW) : VisualWork, MusicalWork, WrittenWork, ArtWork, Software
    Group (GRP) : MusicalGRP, PublicCORP, PrivateCORP, AerospaceManufacturer, SportsGRP, CarManufacturer, ORG
    Person (PER) : Scientist, Artist, Athlete, Politician, Cleric, SportsManager, OtherPER
    Product (PROD) : Clothing, Vehicle, Food, Drink, OtherPROD
    Medical (MED) : Medication/Vaccine, MedicalProcedure, AnatomicalStructure, Symptom, Disease

**Example**
>English: [wes anderson | Artist]'s film [the grand budapest hotel | VisualWork] opened the festival .

>Ukrainian: назва альбому походить з роману « [кінець дитинства | WrittenWork] » англійського письменника [артура кларка | Artist] .

## Approach
Two-stage fine-tuning of Transformer was performed.
### Contrastive learning 
The first stage is a contrastive learning aimed at changing the distance between embeddings of words/sub-words, that was produced by Transformer model. 
For example, named entities of different types have a large distance and small distance for same types. 

This stage based on ideas from [**Contrastive fine-tuning to improve generalization in deep NER**](https://www.dialog-21.ru/media/5751/bondarenkoi113.pdf) (see 3.1 Contrastive fine-tuning)

>You can find SiameseDataset class from *utils/dataset.py* and ContrastiveTrainer class from *trainer.py*

### Fine-tuned BERT + Conditional Random Field  (CoBertCRF)
The second stage is a learning fine-tuned BERT model with CRF from first stage for token classification task (NER). 

