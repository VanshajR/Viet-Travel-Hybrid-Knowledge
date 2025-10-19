# load_to_neo4j.py
import json
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from tqdm import tqdm
import config

DATA_FILE = "vietnam_travel_dataset.json"

# Test connection first
try:
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("✓ Connected to Neo4j successfully")
except AuthError:
    print("✗ Authentication failed. Check NEO4J_USER and NEO4J_PASSWORD in config.py")
    exit(1)
except ServiceUnavailable:
    print(f"✗ Neo4j service unavailable at {config.NEO4J_URI}. Make sure Neo4j is running.")
    exit(1)
except Exception as e:
    print(f"✗ Error connecting to Neo4j: {e}")
    exit(1)

def create_constraints(tx):
    # generic uniqueness constraint on id for node label Entity (we also add label specific types)
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

def upsert_node(tx, node):
    # use label from node['type'] and always add :Entity label
    labels = [node.get("type","Unknown"), "Entity"]
    label_cypher = ":" + ":".join(labels)
    # keep a subset of properties to store (avoid storing huge nested objects)
    props = {k:v for k,v in node.items() if k not in ("connections",)}
    
    # Convert lists to strings for Neo4j compatibility
    if "tags" in props and isinstance(props["tags"], list):
        props["tags"] = ",".join(props["tags"])
    
    # set properties using parameters
    try:
        tx.run(
            f"MERGE (n{label_cypher} {{id: $id}}) "
            "SET n += $props",
            id=node["id"], props=props
        )
    except Exception as e:
        print(f"Error upserting node {node.get('id')}: {e}")

def create_relationship(tx, source_id, rel):
    # rel is like {"relation": "Located_In", "target": "city_hanoi"}
    rel_type = rel.get("relation", "RELATED_TO")
    target_id = rel.get("target")
    if not target_id:
        return
    # Create relationship if both nodes exist
    cypher = (
        "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        "RETURN r"
    )
    tx.run(cypher, source_id=source_id, target_id=target_id)

def main():
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    print(f"✓ Loaded {len(nodes)} nodes from dataset")

    with driver.session() as session:
        session.execute_write(create_constraints)
        
        # Upsert all nodes
        print("Creating/updating nodes...")
        for node in tqdm(nodes, desc="Nodes"):
            session.execute_write(upsert_node, node)

        # Create relationships
        print("Creating relationships...")
        for node in tqdm(nodes, desc="Relationships"):
            conns = node.get("connections", [])
            for rel in conns:
                session.execute_write(create_relationship, node["id"], rel)

    # Verify data load
    with driver.session() as session:
        node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        
        print(f"\n✓ Done loading into Neo4j!")
        print(f"  Nodes in database: {node_count}")
        print(f"  Relationships: {rel_count}")
    
    driver.close()

if __name__ == "__main__":
    main()
