# KGST - Group 3

Files prefixed with `(<some_number>)` are for extracting the information from the research papers.

`conceptnet_relations.py` is used to calculate the similarity score (degrees of relatedness).

`generate_instances_from_extracted_information.ipynb` with `rest_regex.txt` and
`scenario_and_task_regex.txt` is used to generate the `.ttl` file with the
final individuals.

## To run the scripts

You need to
download https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
and extract it to the root of the repository to run the notebook
(from https://github.com/commonsense/conceptnet-numberbatch).