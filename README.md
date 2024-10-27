# Contextual-Multi-Armed-Bandit-To-Download-Webpages

Building a contextual multi-armed bandit (CMAB) agent will downloade Wikipedia webpages related to a given input
subject considering limited webpage storage. Conceptually, the agent will be composed of at least two sub-systems:
– a webpage crawler that explores the Wikipedia links (uri’s)
– CMAB with two actions (arms) that choose to either download a page or not based on the cosine similarity between user input and a selected URI.
