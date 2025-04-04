import requests
from rdflib import Graph, URIRef, RDF
from collections import deque
import time

########################
# CONFIGURATION
########################

ONTOLOGY_FILE = "ontology file path name"  # Path to your TTL file
CONCEPTNET_PREFIX = "https://api.conceptnet.io/c/en/"

MAX_DEPTH = 6             # Maximum depth for BFS shortest path search
BFS_TIMEOUT = 30          # BFS search timeout in seconds
REQUEST_SLEEP_SECS = 0.2  # Delay between API requests to avoid rate-limiting
REQUEST_TIMEOUT = 10      # Timeout in seconds for each API call
MAX_NEIGHBORS = 50        # Max neighbors to consider per concept in BFS
ALLOWED_RELATIONS = {"/r/RelatedTo", "/r/Synonym", "/r/IsA"}  # Relations to traverse in ConceptNet

########################
# LOAD ONTOLOGY
########################

g = Graph()
g.parse(ONTOLOGY_FILE, format="turtle")

# Mapping from instance to list of ConceptNet concept URIs (normalized)
instance_to_concepts = {}
# RDF predicate for ConceptNet subject links
DCTERMS_SUBJECT = URIRef("http://purl.org/dc/terms/subject")

def normalize_concept_uri(uri):
    """
    Strip out 'api.conceptnet.io' if present to get the ConceptNet path,
    and remove any POS/sense tags (e.g., /c/en/dog/n -> /c/en/dog).
    """
    uri_str = str(uri)
    if uri_str.startswith("http"):
        # Keep only the path part after the domain
        uri_str = uri_str.split("api.conceptnet.io")[-1]  # e.g. '/c/en/dog/n'
    parts = uri_str.split('/')
    if len(parts) > 3 and '/' in parts[3]:
        # Remove anything after a slash in the term part (POS tag)
        parts[3] = parts[3].split('/')[0]
    return '/'.join(parts[:4])  # e.g. '/c/en/dog'

# Populate the mapping of instances to ConceptNet terms
for subj, obj in g.subject_objects(DCTERMS_SUBJECT):
    if isinstance(obj, URIRef) and str(obj).startswith(CONCEPTNET_PREFIX):
        norm = normalize_concept_uri(obj)  # normalize the ConceptNet URI
        if subj not in instance_to_concepts:
            instance_to_concepts[subj] = []
        if norm not in instance_to_concepts[subj]:
            instance_to_concepts[subj].append(norm)

########################
# NEIGHBOR CACHING
########################

neighbors_cache = {}

def get_concept_neighbors(concept_uri):
    """
    Retrieve neighboring concept URIs for a given concept (undirected edges),
    considering only allowed relations (RelatedTo, Synonym, IsA).
    Uses caching to avoid duplicate API calls and limits the number of neighbors.
    """
    normalized = normalize_concept_uri(concept_uri)
    if normalized in neighbors_cache:
        return neighbors_cache[normalized]
    time.sleep(REQUEST_SLEEP_SECS)
    url = "https://api.conceptnet.io/query"
    params = {"node": normalized, "limit": 1000}
    neighbors = set()
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        data = resp.json()
        for edge in data.get('edges', []):
            rel_id = edge.get('rel', {}).get('@id')
            if rel_id not in ALLOWED_RELATIONS:
                continue
            start = edge.get('start', {}).get('@id')
            end  = edge.get('end', {}).get('@id')
            # Add the opposite node of the edge if it’s an English concept
            if start == normalized and end and end.startswith("/c/en/"):
                neighbors.add(normalize_concept_uri(end))
            elif end == normalized and start and start.startswith("/c/en/"):
                neighbors.add(normalize_concept_uri(start))
        neighbor_list = list(neighbors)[:MAX_NEIGHBORS]
        neighbors_cache[normalized] = neighbor_list
        return neighbor_list
    except requests.exceptions.Timeout:
        print(f"Timeout fetching neighbors for {normalized}")
    except Exception as e:
        print(f"Error fetching neighbors for {normalized}: {e}")
    # Cache empty result on failure to avoid retrying immediately
    neighbors_cache[normalized] = []
    return []

########################
# BFS SHORTEST PATH
########################

def find_shortest_path(concept_uri1, concept_uri2, max_depth=MAX_DEPTH):
    """
    Perform a breadth-first search (up to max_depth steps) to find the shortest concept chain 
    connecting concept_uri1 to concept_uri2.
    Returns a list of concept URIs (including start and end) if a path is found, or None if no path is found.
    Aborts if the search exceeds BFS_TIMEOUT seconds.
    """
    start = normalize_concept_uri(concept_uri1)
    goal  = normalize_concept_uri(concept_uri2)
    if start == goal:
        return [start]  # Trivial case: same concept
    visited = {start}
    queue = deque([(start, [start])])
    depth = 0
    start_time = time.time()
    # BFS traversal
    while queue and depth <= max_depth:
        if time.time() - start_time > BFS_TIMEOUT:
            return None  # timeout
        level_size = len(queue)
        for _ in range(level_size):
            current, path = queue.popleft()
            for nb in get_concept_neighbors(current):
                if nb not in visited:
                    visited.add(nb)
                    new_path = path + [nb]
                    if nb == goal:
                        return new_path
                    if time.time() - start_time > BFS_TIMEOUT:
                        return None
                    queue.append((nb, new_path))
        depth += 1
    return None  # no path within max_depth

########################
# RELATEDNESS SCORE
########################

def conceptnet_relatedness(concept_uri1, concept_uri2):
    """
    Query ConceptNet's /relatedness endpoint to get a semantic relatedness score between two concepts.
    Returns a score in [-1, 1], or 0.0 on failure.
    """
    c1 = normalize_concept_uri(concept_uri1)
    c2 = normalize_concept_uri(concept_uri2)
    time.sleep(REQUEST_SLEEP_SECS)
    url = "https://api.conceptnet.io/relatedness"
    params = {'node1': c1, 'node2': c2}
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        data = resp.json()
        return data.get("value", 0.0)
    except requests.exceptions.Timeout:
        print(f"Timeout on relatedness for {c1} vs {c2}")
        return 0.0
    except Exception as e:
        print(f"Error fetching relatedness for {c1} vs {c2}: {e}")
        return 0.0

########################
# UTILITIES
########################

def short_name(uri):
    """
    Extract the local name from a URI (portion after the last '#' or '/').
    """
    uri_str = str(uri)
    if '#' in uri_str:
        return uri_str.split('#')[-1]
    else:
        return uri_str.split('/')[-1]

########################
# MAIN
########################

def main():
    user_input = input("Enter the instance IRI (or local name) to analyze: ").strip()
    # Remove angle brackets from input if present (e.g., <...>)
    if user_input.startswith("<") and user_input.endswith(">"):
        user_input = user_input[1:-1]
    
    # Prepare list of all instances that have ConceptNet term mappings
    all_instances = list(instance_to_concepts.keys())
    target_instance = None

    # If the input is a full URI, try to match directly
    if user_input.startswith("http"):
        candidate = URIRef(user_input)
        if candidate in instance_to_concepts:
            target_instance = candidate

    # If not found yet, try matching by local name (case-insensitive)
    if target_instance is None:
        # First, look for an exact local name match
        matches = [inst for inst in all_instances if short_name(inst).lower() == user_input.lower()]
        if not matches:
            # If no exact match, try substring match on local names
            matches = [inst for inst in all_instances if user_input.lower() in short_name(inst).lower()]
        if not matches:
            print("No instance found. Available instances include:")
            for inst in sorted(all_instances, key=lambda x: short_name(x)):
                print("  ", short_name(inst))
            raise ValueError(f"No instance found for '{user_input}'")
        target_instance = matches[0]

    # Retrieve ConceptNet terms for the target instance
    target_concepts = instance_to_concepts.get(target_instance, [])
    if not target_concepts:
        raise ValueError("Selected instance has no ConceptNet terms linked via dcterms:subject.")

    # Display selected instance and its ConceptNet terms
    print(f"\nSelected Instance: {short_name(target_instance)}")
    print(f"ConceptNet terms: {', '.join(short_name(term) for term in target_concepts)}\n")

    # Compare target instance's ConceptNet terms to those of all other instances
    other_instances = sorted([inst for inst in all_instances if inst != target_instance],
                              key=lambda x: short_name(x))
    results = []
    for inst in other_instances:
        other_terms = instance_to_concepts.get(inst, [])
        if not other_terms:
            continue
        # For each combination of target term vs. other term
        for t_concept in target_concepts:
            for o_concept in other_terms:
                score = conceptnet_relatedness(t_concept, o_concept)
                path = find_shortest_path(t_concept, o_concept, max_depth=MAX_DEPTH)
                path_str = " -> ".join(path) if path else f"No path found within depth {MAX_DEPTH}"
                results.append((inst, t_concept, o_concept, score, path_str))

    # Print results in a table format
    print("=== Results ===\n")
    header = "{:<30}  {:<30}  {:<35}  {:>8}  {}"
    print(header.format("Target Instance", "Compared Instance", "Concept Pair", "Score", "Shortest Path"))
    print("-" * 180)
    target_name = short_name(target_instance)
    for inst, t_concept, o_concept, score, path_str in results:
        concept_pair = f"{short_name(t_concept)} vs {short_name(o_concept)}"
        print("{:<30}  {:<30}  {:<35}  {:>8.4f}  {}".format(
            target_name,
            short_name(inst),
            concept_pair,
            score,
            path_str
        ))

if __name__ == "__main__":
    main()