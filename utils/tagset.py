import numpy as np

tags = ['Facility', 'OtherLOC', 'HumanSettlement', 'Station',
        'VisualWork', 'MusicalWork', 'WrittenWork', 'ArtWork', 'Software',
        'OtherCW', 'MusicalGRP', 'PublicCORP', 'PrivateCORP', 'OtherCORP',
        'AerospaceManufacturer', 'SportsGRP', 'CarManufacturer', 'TechCORP',
        'ORG', 'Scientist', 'Artist', 'Athlete', 'Politician', 'Cleric',
        'SportsManager', 'OtherPER', 'Clothing', 'Vehicle', 'Food',
        'Drink', 'OtherPROD', 'Medication/Vaccine', 'MedicalProcedure',
        'AnatomicalStructure', 'Symptom', 'Disease']


def get_tagset():
    # create pairs B-tag and I-tag from fine-grainder tagset of MultiCoNER
    iob_tags = ['O'] + list(np.array([[f'B-{tag}', f'I-{tag}'] for tag in tags]).flatten())
    return dict(zip(iob_tags, range(len(iob_tags))))
