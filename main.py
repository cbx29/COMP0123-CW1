
import sys
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import re
import scipy.stats as stats
from scipy.stats import spearmanr
import subprocess
from pathlib import Path
from typing import Dict, Set, Optional
import seaborn as sns
import plotly.express as px
import numpy as np
import networkx as nx


def filter_and_extract_json(input_file, output_file, type=None, start_date_str=None, end_date_str=None):
    fields_to_keep = {
        "submission": {"author", "author_created_utc", "author_flair_text", "category", "created_utc", "distinguished" "num_comments", "selftext", "title", "id", "ups", "downs", "score"},
        "comment": {"author", "body", "created_utc", "id", "name", "parent_id", "link_id", "ups", "downs", "score", "distinguished"}
    }

    if type not in fields_to_keep:
        raise ValueError(f"Invalid type '{type}'. Must be 'comment' or 'submission'.")

    selected_fields = fields_to_keep[type]

    if start_date_str and end_date_str:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    else:
        start_date = end_date = None

    filtered_data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                entry = json.loads(line)
                if start_date and end_date:
                    created_utc = entry.get('created_utc')
                    if created_utc is not None:
                        if isinstance(created_utc, str):
                            created_utc = int(float(created_utc))
                        created_date = datetime.utcfromtimestamp(created_utc)
                        if not (start_date <= created_date <= end_date):
                            continue

                entry = {k: v for k, v in entry.items() if k in selected_fields}
                filtered_data.append(entry)

            except json.JSONDecodeError as e:
                print(f"JSONDecodeError on line {line_num} in {input_file}: {e}")
                continue

    if not filtered_data:
        print(f"No data found matching criteria in file: {input_file}")
        return False

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
    return True


def json_file_to_parquet(input_file, output_file):
    df = pd.read_json(input_file)
    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype(str)
    

    df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"Successfully converted {input_file} to {output_file}")


def get_temporal_network(start_date = "2023-10-01", end_date = "2023-12-31"):
    comments = pd.read_parquet("filtered_parquet/london_comments.parquet")
    submissions = pd.read_parquet("filtered_parquet/london_submissions.parquet")

    start_timestamp = pd.Timestamp(start_date).timestamp()
    end_timestamp = pd.Timestamp(end_date).timestamp()

    submissions = submissions[(submissions["created_utc"] >= start_timestamp) & 
                              (submissions["created_utc"] <= end_timestamp)]

    valid_users = set(submissions["author"].dropna())
    submissions = submissions[~submissions["author"].isin(["[deleted]"])]
    submissions = submissions[~submissions["author"].isin(["AutoModerator"])]
    valid_users = set(submissions["author"])

    submission_author_map = submissions.set_index("id")["author"].to_dict()

    comments = comments[(comments["created_utc"] >= start_timestamp) & 
                        (comments["created_utc"] <= end_timestamp)]

    comments["submission_id"] = comments["link_id"].str.replace("t3_", "")
    comments = comments[comments["submission_id"].isin(submission_author_map)]
    comments = comments[~comments["author"].isin(["[deleted]"])]
    comments = comments[~comments["author"].isin(["AutoModerator"])]
    comments = comments[comments["author"].isin(valid_users)]

    comments["post_author"] = comments["submission_id"].map(submission_author_map)

    comments = comments[(comments["author"] != comments["post_author"]) & 
                        (comments["post_author"].isin(valid_users))]

    interaction_graph = nx.DiGraph()

    edges = comments[["author", "post_author"]].drop_duplicates()
    interaction_graph.add_edges_from(edges.itertuples(index=False, name=None))

    output_filename = f"london_user_interaction_{start_date}_to_{end_date}.gexf"
    nx.write_gexf(interaction_graph, output_filename)
    print(f"Graph has {interaction_graph.number_of_nodes()} nodes and {interaction_graph.number_of_edges()} edges.")
    print(f"Graph saved as {output_filename}.")


def plot_degree_distributions(gexf_file1, gexf_file2):
    G1 = nx.read_gexf(gexf_file1)
    G2 = nx.read_gexf(gexf_file2)

    if not G1.is_directed():
        G1 = G1.to_directed()
    if not G2.is_directed():
        G2 = G2.to_directed()

    in_degrees1 = [deg for _, deg in G1.in_degree()]
    out_degrees1 = [deg for _, deg in G1.out_degree()]
    in_degrees2 = [deg for _, deg in G2.in_degree()]
    out_degrees2 = [deg for _, deg in G2.out_degree()]

    def plot_single_distribution(ax, degrees1, degrees2, degree_type, color):
        unique_degrees1, counts1 = np.unique(degrees1, return_counts=True)
        probabilities1 = counts1 / counts1.sum()

        log_degrees1 = np.log10(unique_degrees1[unique_degrees1 > 0])
        log_probabilities1 = np.log10(probabilities1[unique_degrees1 > 0])

        slope, intercept = np.polyfit(log_degrees1, log_probabilities1, 1)

        unique_degrees2, counts2 = np.unique(degrees2, return_counts=True)
        probabilities2 = counts2 / counts2.sum()

        ax.loglog(unique_degrees1, probabilities1, 'o', markersize=5, alpha=0.5, color=color, label=f"r/london ({degree_type})")
        ax.loglog(unique_degrees2, probabilities2, 'o', markersize=5, alpha=0.5, color='green', label=f"ER ({degree_type})")

        ax.plot(unique_degrees1[unique_degrees1 > 0], 10**(slope * log_degrees1 + intercept), '--', color=color, label=f"r/london Best Fit (Slope = {slope:.2f})")

        ax.set_xlabel("Degree k")
        ax.set_ylabel("Degree Probability P(k)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    plot_single_distribution(axs[0], in_degrees1, in_degrees2, "In-Degree", "blue")
    axs[0].legend()
    axs[0].set_title("In-Degree Distribution (Log-Log Scale)")

    plot_single_distribution(axs[1], out_degrees1, out_degrees2, "Out-Degree", "blue")
    axs[1].legend()
    axs[1].set_title("Out-Degree Distribution (Log-Log Scale)")

    plt.suptitle("Degree Distributions of r/london with ER for Comparison", fontsize=16)
    plt.show()


def pagerank(gexf_file, top_n=10):
    G = nx.read_gexf(gexf_file)

    pagerank_scores = nx.pagerank(G)

    sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    top_users = sorted_pagerank[:top_n]


    df_top_users = pd.DataFrame(top_users, columns=["User", "PageRank Score"])

    plt.figure(figsize=(8, 6))
    plt.barh(df_top_users["User"], df_top_users["PageRank Score"], color='skyblue')
    plt.title(f"Top {top_n} Influential Users by PageRank", fontsize=16)
    plt.xlabel("PageRank Score", fontsize=12)
    plt.ylabel("Users", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return df_top_users


def randomize_edges_and_save(gexf_file, output_file):
    G = nx.read_gexf(gexf_file)
    
    is_directed = G.is_directed()
    
    if is_directed:
        randomized_G = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), directed=True)
    else:
        randomized_G = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), directed=False)
    
    for node in G.nodes(data=True):
        if node[0] in randomized_G.nodes:
            nx.set_node_attributes(randomized_G, {node[0]: node[1]})
    
    nx.write_gexf(randomized_G, output_file)
    print(f"Graph has {randomized_G.number_of_nodes()} nodes and {randomized_G.number_of_edges()} edges.")
    print(f"Randomized graph saved to: {output_file}")



if __name__ == "__main__":
    # filter_and_extract_json("/Users/baixu/Documents/UCL/Year-4/COMP0123/CW1/reddit/raw_json/london_submissions.ndjson", "filtered_json/london_submissions.json", "submission")
    # json_file_to_parquet("filtered_json/london_comments.json", "london_comments.parquet")
    # get_temporal_network()
    # randomize_edges_and_save("london_user_interaction_2023-10-01_to_2023-12-31.gexf", "random.gexf")
    # plot_degree_distributions("london_user_interaction_2023-10-01_to_2023-12-31.gexf", "random.gexf")
    pagerank("london_user_interaction_2023-10-01_to_2023-12-31.gexf", 50)
