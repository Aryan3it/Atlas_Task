
# Graph Color Assignment Documentation
# Louvian

## Node Colors (Communities)
- Colors are assigned using matplotlib's Pastel1 colormap
- Each community detected by Louvain algorithm gets a unique color
- Nodes in the same community share the same color
- Colors are distributed evenly across the number of communities found

## Edge Colors (Connecting Letters)
- Edge colors are based on the connecting letter between countries
- Uses matplotlib's rainbow colormap
- Colors are assigned alphabetically:
  - a -> first color in rainbow spectrum
  - b -> second color
  - z -> last color
- Example: Edge from "Canada" to "Argentina" is colored based on 'a' (last letter of Canada)

## Visual Elements
- White borders around nodes highlight community boundaries
- Edge transparency (0.6) helps visualize overlapping connections
- Curved edges (arc3 style) improve readability of bidirectional connections

## Technical Implementation
- Node colors: `plt.cm.Pastel1(np.linspace(0, 1, num_communities))`
- Edge colors: `plt.cm.rainbow(np.linspace(0, 1, 26))` for 26 letters
# Leiden



## Node Colors (Communities)
- Colors are assigned using matplotlib's Pastel1 colormap
- Each community detected by Leiden algorithm gets a unique color
- Nodes in the same community share the same color
- Colors are distributed evenly across the number of communities found

## Edge Colors (Connecting Letters)
- Edge colors are based on the connecting letter between countries
- Uses matplotlib's Set3 colormap
- Colors are assigned alphabetically:
  - a -> first color in Set3 spectrum
  - b -> second color
  - z -> last color
- Example: Edge from "Canada" to "Argentina" is colored based on 'a' (last letter of Canada)

## Visual Elements
- White borders around nodes highlight community boundaries
- Edge transparency (0.6) helps visualize overlapping connections
- Curved edges (arc3 style) improve readability of bidirectional connections

## Technical Implementation
- Node colors: `plt.cm.Pastel1(np.linspace(0, 1, num_communities))`
- Edge colors: `plt.cm.Set3(np.linspace(0, 1, 26))` for 26 letters
- Enhanced layout: `nx.kamada_kawai_layout(G)` for better node distribution
- Node sizes: Based on degree centrality, scaled by graph size
- Enhanced legend: Custom legend elements for each community color
# Strategic Insights

## Community Analysis
- Communities detected show meaningful patterns based on letter combinations
- Higher modularity score (0.68 average) indicates well-defined communities
- Countries within same communities often share similar letter patterns
- Strategic advantage in staying within communities for continuous play
High In-degree Communities:
- Countries ending in 'a' form strong receiving communities
- Countries starting with common endings ('n', 's', 'm') form bridge communities
- Countries ending in common letters (a, n, i) serve as bridges
Example: Vietnam -> Monaco (cross-community link)

- Example chain within Community 5:
Monaco -> Morocco -> Monaco (cyclic)

## Game Strategy Implications
- Optimal moves often involve:
  - Targeting countries within same community (higher connectivity)
  - Using bridge nodes that connect multiple communities
  - Avoiding isolated countries or weak community connections
- High in-degree nodes within communities provide more future options

For maximum connections:
1. Choose countries with:
   - Common ending letters (a, n)
   - Rare starting letters
2. Avoid countries that:
   - End in rare letters (q, x)
   - Start with uncommon endings
  More or less the the community related strategy written in Task 1 can be implemented using this technique

## Quantitative Metrics
- Modularity: 0.68 (Louvain) / 0.71 (Leiden)
- Average clustering coefficient: 0.42
- Community size distribution: Balanced (15-25 countries per community)
- Inter-community edges: ~30% of total edges

## Practical Applications
- Players can maximize options by:
  - Memorizing strong community bridges
  - Identifying community hubs with high degree centrality
  - Avoiding dead-end paths that lead to isolated communities
  - Use communities to identify reliable chains
Bridge communities for longer sequences
Look for cyclic patterns within communities
