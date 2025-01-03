# earlyDETECT
A simple project demonstrating a deployed machine learning system for prescreening for the likelihood of depression.

### Description
In this project, we demonstrate a possible workflow for prescreening for depression based on a trained ML model and five features. These features include:
- `Average number of characters in response`
- `Average number of nouns used in the response`
- `Speech speed`
- Sentiment of response to `How are you at controlling your temper?`
- Sentiment of response to `When was the last time you argued with someone and what was it about?`

We also provide a possible explanation for the model's decision in terms of feature impact or contribution to every outcome. For each interpretation, we leverage LIME (Local Interpretable Model-Agnostic Explanations). After each inference, users can click on the visualization dropdown in the top right to view the LIME plots, Feature importance, and one of the decision trees.

To watch the demo, click on the YouTube icon in the first image.  
[![Watch the demo video](./for_readme/interface_1.png)](http://www.youtube.com/watch?v=k5R3xtf2gWU')
![DDS_0](./for_readme/interface_2.png)
![DDS_1](https://github.com/MustaphaU/earlyDETECT/assets/123378149/74e67d9d-ea39-4397-8895-c1452b944652)
![DDS_2](https://github.com/MustaphaU/earlyDETECT/assets/123378149/ba5a8ce7-ab84-42dd-97c7-415135ba0989)

Here is the UML diagram:
![DDS_3](./for_readme/user_interaction.png)

The architecture diagram is:
![DDS_4](./for_readme/architecture_dds.png)
