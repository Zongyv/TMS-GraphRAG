"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""
RESPONSE_IN_MARKDOWN__ = """---Role---

You are a professional medical assistant. Please answer the questions in concise and accurate medical language and avoid using lists and numbers.

---Requirements---
1. Answer only based on the provided medical knowledge
2. Use professional terms but keep them easy to understand
3. The format of the answer should be consecutive paragraphs and subheadings should be avoided
4. When in doubt, clearly state "It cannot be determined based on the available information.

---Target response length and format---

{response_type}


---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""
RAG_RESPONSE = "local_rag_response"

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

# 帮助分析人员从文本中提取与特定实体相关的负面或风险声明（claims）信息
PROMPTS[
    "claim_extraction"
] = """-Target activity-
You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

-Goal-
Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.
For each claim, extract the following information:
- Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
- Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
- Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
- Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
- Claim Description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references.
- Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
- Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

Format each claim as (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Examples-
Example 1:
Entity specification: organization
Claim description: red flags associated with an entity
Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
Output:

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
{completion_delimiter}

Example 2:
Entity specification: Company A, Person C
Claim description: red flags associated with an entity
Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
Output:

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
{record_delimiter}
(PERSON C{tuple_delimiter}NONE{tuple_delimiter}CORRUPTION{tuple_delimiter}SUSPECTED{tuple_delimiter}2015-01-01T00:00:00{tuple_delimiter}2015-12-30T00:00:00{tuple_delimiter}Person C was suspected of engaging in corruption activities in 2015{tuple_delimiter}The company is owned by Person C who was suspected of engaging in corruption activities in 2015)
{completion_delimiter}

-Real Data-
Use the following input for your answer.
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output: """


# 社区报告提示词，用以引导生成结构化的社区报告
PROMPTS[
    "community_report"
] = """You are an AI assistant that helps a human analyst to perform general information discovery. 
Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:
```
Entities:
```csv
id,entity,type,description
5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,organization,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
```
Relationships:
```csv
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
```
```
Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes."
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved."
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.

Output:
"""


# # 从文本中识别特定类型的实体和它们之间的关系，并以结构化格式输出。
# PROMPTS[
#     "entity_extraction"
# ] = """-Goal-
# Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
#
# -Steps-
# 1. Identify all entities. For each identified entity, extract the following information:
# - entity_name: Name of the entity, capitalized
# - entity_type: One of the following types: [{entity_types}]
# - entity_description: Comprehensive description of the entity's attributes and activities
# Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
#
# 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
# For each pair of related entities, extract the following information:
# - source_entity: name of the source entity, as identified in step 1
# - target_entity: name of the target entity, as identified in step 1
# - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
# - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
#  Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
#
# 3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.
#
# 4. When finished, output {completion_delimiter}
#
# ######################
# -Examples-
# ######################
# Example 1:
#
# Entity_types: [person, technology, mission, organization, location]
# Text:
# while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
#
# Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”
#
# The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
#
# It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
# ################
# Output:
# ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
# ("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
# ("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
# ("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
# ("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
# ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
# ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
# #############################
# Example 2:
#
# Entity_types: [person, technology, mission, organization, location]
# Text:
# They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.
#
# Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.
#
# Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
# #############
# Output:
# ("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
# ("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
# ("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
# ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
# ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
# #############################
# Example 3:
#
# Entity_types: [person, role, technology, organization, event, location, concept]
# Text:
# their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.
#
# "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."
#
# Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."
#
# Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.
#
# The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
# #############
# Output:
# ("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
# ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
# ("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
# ("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
# ("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
# ("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
# ("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
# #############################
# -Real Data-
# ######################
# Entity_types: {entity_types}
# Text: {input_text}
# ######################
# Output:
# """

PROMPTS[
    "entity_extraction"
] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1.
- target_entity: name of the target entity, as identified in step 1.
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Requirement-
1. For all extracted entities, including "entity_name", "source_entity" and "target_entity", if this entity is an abbreviation of a proper noun, the full name is used instead, and the abbreviation no longer needs to be provided in parentheses.
For example, regarding "TMS", "transcranial magnetic stimulation", and "transcranial magnetic stimulation (TMS)", Change it to "TRANSCRANIAL MAGNETIC STIMULATION" when extracting the entity.

2. If the extracted entity contains parentheses, delete the parentheses and the contents within them.
For example, "MOTOR EVOKED POTENTIAL (MEP) AMPLITUDES" is modified to "MOTOR EVOKED POTENTIAL AMPLITUDES".

######################
-Examples-
######################
Example 1:

Entity_types: [disease name, stimulate brain region, stimulus parameters, evaluation tools and scales, transcranial magnetic stimulation equipment, stimulating coil type, therapeutic outcome]
Text:
Objective: The aim of this study was to evaluate the effects of transcranial magnetic stimulation synchronized with maximal effort to make a target movement in patients with chronic hemiplegia involving the hand.
Design: Non-randomized double-blinded controlled trial.
Subjects: Nine chronic patients with hemiplegia who were unable to fully extend the affected fingers following stroke. Methods: Patients were assigned to receive 100 pulses of active or sham transcranial magnetic stimulation of the affected hemisphere per session. Each active or sham pulse was delivered during maximal effort at thumb and finger extension as a target movement. A blinded rater assessed stroke impairments at baseline, immediately after, and one week after 4 weekly transcranial magnetic stimulation sessions. Motor evoked potential amplitudes were measured at each session.
Results: All sessions were completed without adverse effects. Immediately after the fourth transcranial magnetic stimulation session, 4 of 5 patients in the active transcranial magnetic stimulation group (80%) had either reduced wrist flexor spasticity or improved manual performance; no such change occurred in the sham group (Fisher’s exact test, p < 0.05). Effects persisted one week later. In the active transcranial magnetic stimulation group, 3 patients who showed an increase in motor evoked potential amplitudes all had improvement in clinical assessments.
Conclusion: Transcranial magnetic stimulation synchronized with maximum effort to make a target movement improved hand motor function in patients with chronic hemiplegia.
################
Output:
("entity"{tuple_delimiter}"Transcranial Magnetic Stimulation"{tuple_delimiter}"transcranial magnetic stimulation equipment"{tuple_delimiter}"Transcranial Magnetic Stimulation (TMS) is a therapeutic technique used to stimulate brain regions, enhancing motor function in patients with hemiplegia."){record_delimiter}
("entity"{tuple_delimiter}"Hemiplegia"{tuple_delimiter}"disease name"{tuple_delimiter}"Hemiplegia is a condition causing paralysis of one side of the body, often resulting from a stroke."){record_delimiter}
("entity"{tuple_delimiter}"100 pulses per session, 4 weekly sessions"{tuple_delimiter}"stimulus parameters"{tuple_delimiter}"The specific parameters of the TMS protocol, including the number of pulses and frequency of sessions."){record_delimiter}
("entity"{tuple_delimiter}"Affected Hemisphere"{tuple_delimiter}"stimulate brain region"{tuple_delimiter}"The brain region targeted by TMS, typically the motor area of the hemisphere affected by stroke."){record_delimiter}
("entity"{tuple_delimiter}"Motor Evoked Potential Amplitudes"{tuple_delimiter}"evaluation  tools and  scales"{tuple_delimiter}"Motor Evoked Potential (MEP)  amplitudes are measured to assess changes in motor function following TMS sessions."){record_delimiter}
("entity"{tuple_delimiter}"Blinded Clinical Assessment"{tuple_delimiter}"evaluation tools and scales"{tuple_delimiter}"A blinded  rater assesses stroke impairments at baseline, immediately after, and one week after TMS sessions."){record_delimiter}
("entity"{tuple_delimiter}"Reduced Wrist Flexor Spasticity"{tuple_delimiter}"therapeutic outcome"{tuple_delimiter}"A therapeutic outcome observed in 80% of active TMS group patients, indicating improved motor function."){record_delimiter}
("entity"{tuple_delimiter}"Improved Manual Performance"{tuple_delimiter}"therapeutic outcome"{tuple_delimiter}"Patients showed improvement in manual performance following TMS sessions, enhancing their motor capabilities."){record_delimiter}
("entity"{tuple_delimiter}"Motor Function Improvement"{tuple_delimiter}"therapeutic outcome"{tuple_delimiter}"Overall improvement in motor function was observed in patients with chronic hemiplegia after TMS treatment."){record_delimiter}
("relationship"{tuple_delimiter}"Transcranial Magnetic Stimulation"{tuple_delimiter}"Affected Hemisphere"{tuple_delimiter}"Transcranial magnetic stimulation is applied to the affected hemisphere to modulate cortical excitability and improve motor function."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Transcranial Magnetic Stimulation"{tuple_delimiter}"100 pulses per session, 4 weekly sessions"{tuple_delimiter}"The stimulus parameters define how transcranial magnetic stimulation is administered in the study."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Motor Evoked Potential Amplitudes"{tuple_delimiter}"Reduced Wrist Flexor Spasticity"{tuple_delimiter}"MEP amplitudes are used to evaluate the therapeutic outcome of reduced spasticity in patients."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Blinded Clinical Assessment"{tuple_delimiter}"Improved Manual Performance"{tuple_delimiter}"Clinical assessments measure the therapeutic outcome of improved  manual performance in  patients."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Transcranial Magnetic Stimulation"{tuple_delimiter}"Motor Function Improvement"{tuple_delimiter}"TMS leads to significant motor function improvement in patients with chronic hemiplegia."{tuple_delimiter}9){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""



PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

# 用于在初次实体提取结果遗漏过多实体的情况下，继续提取遗漏的实体。
PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

# 这是一个用于判断是否仍有遗漏实体的提示语句。系统或分析流程会在多轮实体提取后使用此提示，来询问是否还有剩余的、未被提取的实体。
PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

# 默认实体类型列表，用于实体提取提示词中。
# PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["disease name", "stimulate brain region", "stimulus parameters", "evaluation tools and scales", "transcranial magnetic stimulation equipment", "stimulating coil type", "therapeutic outcome"]

# 默认元组分隔符，用于实体提取提示词中。
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"

# 默认记录分隔符，用于实体提取提示词中。
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"

# 默认完成分隔符，用于实体提取提示词中。
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"


# 根据用户提问和表格数据生成准确、格式化的回答。
# PROMPTS[
#     "local_rag_response"
# ] = """---Role---
#
# You are a helpful assistant responding to questions about data in the tables provided.
#
#
# ---Goal---
#
# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
# If you don't know the answer, just say so. Do not make anything up.
# Do not include information where the supporting evidence for it is not provided.
#
# ---Target response length and format---
#
# {response_type}
#
#
# ---Data tables---
#
# {context_data}
#
#
# ---Goal---
#
# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
#
# If you don't know the answer, just say so. Do not make anything up.
#
# Do not include information where the supporting evidence for it is not provided.
#
#
# ---Target response length and format---
#
# {response_type}
#
# Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
# """

PROMPTS[
    "local_rag_response"
] = """%s""" % RESPONSE_IN_MARKDOWN__



# 在 使用表格数据作为输入上下文 的情况下，生成一个 JSON 格式的“关键要点列表”，每个要点都包含详细描述和重要性评分。
PROMPTS[
    "global_map_rag_points"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1...", "score": score_value}},
        {{"description": "Description of point 2...", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}
"""

# 从多个分析师的报告中综合信息，生成一个全面的回答。
PROMPTS[
    "global_reduce_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

# 一个简单的 RAG 响应生成器，用于在 使用文本数据作为输入上下文 的情况下，生成一个回答。
# PROMPTS[
#     "naive_rag_response"
# ] = """You are a professional medical assistant and a master of transcranial magnetic stimulation technology. Please answer the questions in concise and accurate medical language and avoid using lists and numbers.
# Below are the knowledge you know:
# {content_data}
# ---Goal---
# If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
# If you don't know the answer, just say so. Do not make anything up.
# Do not include information where the supporting evidence for it is not provided.
# ---Target response length and format---
# {response_type}
# """

PROMPTS[
    "naive_rag_response"
] = """You are a professional medical assistant and a master of transcranial magnetic stimulation technology. Please answer the questions in concise and accurate medical language and avoid using lists and numbers.
Below are the knowledge you know:
{content_data}

---Requirements---
1. Answer only based on the provided medical knowledge
2. Use professional terms but keep them easy to understand
3. The format of the answer should be consecutive paragraphs and subheadings should be avoided
4. When in doubt, clearly state "It cannot be determined based on the available information.

---Target response length and format---
{response_type}
"""

PROMPTS["metadata_prompt"] = """
从以下论文片段提取meta分析基础信息，返回JSON格式：

{combined_content}

提取信息：
{{
    "title": "论文标题",
    "study_type": "研究类型(RCT/cohort等)",
    "author": "第一作者",
    "year": "发表年份",
    "key_words": "关键词列表",
    "population": "研究人群(MDD/AD等)",
    "sample_size": "总样本量",
    "control_sample_size": "对照组样本量",
    "intervention_sample_size": "干预组样本量",
    "num_follow_up_people": "随访人数",
    "control_intervention": "对照组干预措施",
    "intervention_intervention": "干预组干预措施"
}}

提取规则：
1.只返回JSON，未提及的设为null。
2.重点注意"sample_size"应该为进行随机分组时的样本数，而不是研究开始时的样本数，因为只有被随机分组的才算是进行研究。同时也不是研究结束时的样本数，在摘要中作者有时为了美化数据，会将num_follow_up_people记作sample_size。
3.如果摘要中没有明确的key words信息，则返回null，不要自行提取。

"""

PROMPTS["study_design_identification"] = """
从以下论文内容中识别研究设计类型，返回JSON格式：

{combined_content}

请识别研究设计并返回：

{{
    "study_design_type": "研究设计类型",
    "design_details": {{
        "is_crossover": true/false,
        "is_multi_arm": true/false,
        "number_of_arms": 组数,
        "crossover_periods": 交叉期数(如果是交叉设计),
        "washout_period": "洗脱期时长(如果是交叉设计)",
        "sequence_description": "交叉顺序描述(如AB/BA)"
    }}
}}

研究设计类型"study_design_type"的分类，注意只采用以下分类：
1. "case report" - 案例报告/病例报告
2. "open label" - 开放标签研究
3. "two-arm rct" - 双臂随机对照试验
4. "multi-arm rct" - 多臂随机对照试验(3个及以上组别)
5. "double-blind sham-controlled crossover" - 双盲随机交叉试验

识别要点：
- 交叉设计关键词：crossover, cross-over, AB/BA sequence, washout period
- 多臂设计：明确提到3个或更多组别
- 开放标签：open-label, unblinded
- 案例报告：case report, case series

只返回JSON格式，无数据的字段设为null。
"""


PROMPTS["outcome_prompt"] = """
从以下论文内容中提取结局指标定义，返回JSON格式：

{combined_content}

请识别并提取：

{{
    "primary_outcomes": [
        {{
            "name": "结局指标名称",
            "scale": "量表名称(如HAMD-17, MADRS等)",
            "description": "指标描述",
            "measurement_time": "测量时间点"
        }}
    ],
    "secondary_outcomes": [
        {{
            "name": "结局指标名称", 
            "scale": "量表名称",
            "description": "指标描述",
            "measurement_time": "测量时间点"
        }}
    ]
}}

提取规则：
1. 明确区分主要结局指标(primary outcome)和次要结局指标(secondary outcome)
2. 提取具体的量表名称(如HAMD-17, MADRS, BDI, HAMA等)
3. 记录测量时间点(如baseline, post-treatment, 4weeks等)
4. 只提取文中明确提及的结局指标
5. 在提取次要结局指标时，仅提取你认为最核心的几个次要结局指标
6. 如果论文内容中完全没有提及主要结局指标和次要结局指标这个概念，则将你认为是主要结局指标的指标定义为主要结局指标，将其他指标定义为次要结局指标
7. 返回结果用英文表示

只返回JSON格式，无相关信息设为空数组。
"""

PROMPTS["numerical_prompt"] = """
从表格及其上下文中提取meta分析所需的精确数值数据，返回JSON：

{table_content}

其中，如果表格中有以下的结局指标，则仅提取这些指标的数据：
目标主要结局指标为{primary_outcomes}，次要结局指标为{secondary_outcomes}

请按以下格式提取数据：

{{
    "primary_outcomes": [
        {{
            "outcome_name": "主要结局指标名称",
            "direction": "higher_is_better/ lower_is_better",
            "time_point": "测量时间点(如baseline/4weeks/8weeks等)",
            "intervention_group": {{
                "n": 样本量,
                "mean": 均值,
                "sd": 标准差,
                "ci_lower": 置信区间下限,
                "ci_upper": 置信区间上限,
                "ci_percent": 置信区间百分比,
                "median": 中位数,
                "iqr": [四分位距下限, 四分位距上限]
            }},
            "control_group": {{
                "n": 样本量,
                "mean": 均值,
                "sd": 标准差,
                "ci_lower": 置信区间下限,
                "ci_upper": 置信区间上限,
                "ci_percent": 置信区间百分比,
                "median": 中位数,
                "iqr": [四分位距下限, 四分位距上限]
            }},
            "effect_size": {{
                "value": 效应量数值,
                "type": "效应量类型(SMD/MD/OR/RR等)",
                "ci_lower": 置信区间下限,
                "ci_upper": 置信区间上限,
                "p_value": p值,
                "t_value": t统计量,
                "z_value": z统计量
            }}
        }}
    ],
    "secondary_outcomes": [
        {{
            "outcome_name": "次要结局指标名称",
            "direction": "higher_is_better/ lower_is_better",
            "time_point": "测量时间点",
            "intervention_group": {{
                "n": 样本量,
                "mean": 均值,
                "sd": 标准差,
                "ci_lower": 置信区间下限,
                "ci_upper": 置信区间上限,
                "ci_percent": 置信区间百分比,
            }},
            "control_group": {{
                "n": 样本量,
                "mean": 均值,
                "sd": 标准差,
                "ci_lower": 置信区间下限,
                "ci_upper": 置信区间上限,
                "ci_percent": 置信区间百分比,
            }},
            "effect_size": {{
                "value": 效应量数值,
                "type": "效应量类型",
                "ci_lower": 置信区间下限,
                "ci_upper": 置信区间上限,
                "p_value": p值,
                "t_value": t统计量,
                "z_value": z统计量
            }}
        }}
    ],
    "baseline_characteristics": {{
        "total_sample_size": 总样本量,
        "intervention_n": 干预组样本量,
        "control_n": 对照组样本量,
        "mean_age": 平均年龄,
        "intervention_group_age": 干预组平均年龄,
        "control_group_age": 对照组平均年龄,
        "gender_distribution": {{"male_percent": 男性比例, "female_percent": 女性比例}},
        "intervention_group_gender_distribution": {{
            "male_percent": 干预组男性比例,
            "female_percent": 干预组女性比例
        }},
        "control_group_gender_distribution": {{
            "male_percent": 对照组男性比例,
            "female_percent": 对照组女性比例
        }},
        "dropout_rate": 脱落率
    }}
}}

重要提取规则：
1. 优先提取主要结局指标的完整统计数据(n, mean, SD)
2. 如果某个结局指标有多个测量时间点，则不提取baseline测量时间点的数据，其他数据可以提取
3. 对于连续性变量，必须提取mean和SD；对于分类变量，提取事件数和总数
4. **重要：即使没有直接的SD，也要提取CI、p值、t值等统计量用于后续计算**
5. 置信区间要明确标注百分比(95%CI, 90%CI等)
6. 如果结局指标是记录量表在两个时间节点的变化值，则在量表名称前面加上"change in"
7. 效应量如果论文中已计算，直接提取；如果没有，设为null
8. 优先提取表格中明确显示的数值，如果没有相应数据，再在表格上下文中寻找，如果还没有就不要推测，设为null
9. 对于所给的输入文本，如果发现某项指标的测量时间点的值仅为baseline，则不对该指标进行提取

只返回JSON格式，无数据的字段设为null。
"""
#2. 如果某个结局指标有多个测量时间点，仅提取经过时间最长的测量数据，比如同时出现治疗后和随访数据，仅提取随访数据；同时出现4weeks和8weeks的数据，仅提取8weeks的数据

PROMPTS["numerical_prompt_multi_arm"] = """
从表格及其上下文中提取多臂RCT的数值数据，返回JSON：

{table_content}

目标主要结局指标为{primary_outcomes}，次要结局指标为{secondary_outcomes}

{{
    "number_of_arms": 组数,
    "primary_outcomes": [
        {{
            "outcome_name": "主要结局指标名称",
            "direction": "higher_is_better/lower_is_better",
            "time_point": "测量时间点",
            "groups": [
                {{
                    "group_id": 1,
                    "group_name": "组别名称",
                    "group_type": "control/intervention",
                    "intervention_description": "干预措施描述",
                    "n": 样本量,
                    "mean": 均值,
                    "sd": 标准差,
                    "ci_lower": 置信区间下限,
                    "ci_upper": 置信区间上限,
                    "ci_percent": 置信区间百分比
                }}
            ]
        }}
    ],
    "secondary_outcomes": [
        {{
            "outcome_name": "次要结局指标名称",
            "direction": "higher_is_better/lower_is_better",
            "time_point": "测量时间点",
            "groups": [
                {{
                    "group_id": 1,
                    "group_name": "组别名称",
                    "group_type": "control/intervention",
                    "intervention_description": "干预措施描述",
                    "n": 样本量,
                    "mean": 均值,
                    "sd": 标准差,
                    "ci_lower": 置信区间下限,
                    "ci_upper": 置信区间上限,
                    "ci_percent": 置信区间百分比
                }}
            ]
        }}
    ],
    "baseline_characteristics": {{
        "total_sample_size": 总样本量,
        "groups": [
            {{
                "group_id": 1,
                "n": 样本量,
                "mean_age": 平均年龄,
                "gender_distribution": {{"male_percent": 男性比例, "female_percent": 女性比例}}
            }}
        ]
    }}
}}

注意：
1. 优先提取主要结局指标的完整统计数据(n, mean, SD)
2. 每个组别必须有唯一的group_id
3. 必须明确标注哪个是对照组(group_type: "control")
4. 如果某个结局指标有多个测量时间点，则不提取baseline测量时间点的数据，其他数据可以提取
5. **重要：即使没有直接的SD，也要提取CI、p值、t值等统计量用于后续计算**
6. 如果结局指标是记录量表在两个时间节点的变化值，则在量表名称前面加上"change in"
7. 置信区间要明确标注百分比(95%CI, 90%CI等)
8. 效应量如果论文中已计算，直接提取；如果没有，设为null
9. 优先提取表格中明确显示的数值，如果没有相应数据，再在表格上下文中寻找，如果还没有就不要推测，设为null
10. 对于所给的输入文本，如果发现某项指标的测量时间点的值仅为baseline，则不对该指标进行提取

只返回JSON格式，无数据的字段设为null。
"""

PROMPTS["numerical_prompt_crossover"] = """
从表格及其上下文中提取交叉设计RCT的数值数据，返回JSON：

{table_content}

目标主要结局指标为{primary_outcomes}，次要结局指标为{secondary_outcomes}

{{
    "crossover_details": {{
        "number_of_periods": 期数,
        "sequence_groups": ["AB", "BA"]
    }},
    "primary_outcomes": [
        {{
            "outcome_name": "主要结局指标名称",
            "direction": "higher_is_better/lower_is_better",
            "periods": [
                {{
                    "period_id": 1,
                    "period_name": "Period 1/Period 2",
                    "conditions": [
                        {{
                            "condition_type": "intervention/control",
                            "sequence_group": "AB/BA",
                            "n": 样本量,
                            "mean": 均值,
                            "sd": 标准差,
                            "ci_lower": 置信区间下限,
                            "ci_upper": 置信区间上限
                        }}
                    ]
                }}
            ],
            "carryover_effect": {{
                "tested": true/false,
                "p_value": p值,
                "conclusion": "是否存在携带效应"
            }}
        }}
    ],
    "secondary_outcomes": [...]
}}

注意：
1. 交叉设计需要记录每个period的数据
2. 区分不同sequence组(AB vs BA)
3. 检查是否报告了携带效应(carryover effect)，返回"yes/no"，如果没提到，设为null
4."secondary_outcomes"中返回的格式应该与"primary_outcomes"一致
5. **重要：即使没有直接的SD，也要提取CI、p值、t值等统计量用于后续计算**
6. 如果结局指标是记录量表在两个时间节点的变化值，则在量表名称前面加上"change in"
7. 优先提取表格中明确显示的数值，如果没有相应数据，再在表格上下文中寻找，如果还没有就不要推测，设为null

严格按照上述格式只返回JSON格式，无数据的字段设为null。
"""

PROMPTS["group_selection_prompt"] = """
这是一个多臂RCT研究，包含以下{total_groups}个组别：

{group_descriptions}
{tms_info}
{query_context}

请从这些组别中选择最适合进行Meta分析比较的两组：
1. 一个干预组（intervention group）：通常是接受主要治疗的组
2. 一个对照组（control group）：通常是假刺激组、安慰剂组或标准治疗组

选择标准：
- **优先考虑用户研究问题的关注点**（如特定频率、特定靶点等）
- 优先选择真实刺激和假刺激的对比，但如果用户研究问题中侧重不同刺激方法的对比，则选择不同刺激方法的对比
- 如果有多个真实刺激组，根据用户问题选择最相关的参数，如果没提到参数，则选择最有可能被研究的刺激组
- 确保两组数据完整（有n、mean、sd）

仿照以下json格式，返回结果：
{{
  "intervention_group_id": 1,
  "control_group_id": 3,
  "rationale": "根据用户问题关于10Hz rTMS的研究，选择10Hz rTMS组作为干预组，假刺激组作为对照组"
}}

只返回JSON，不要其他内容。
"""


PROMPTS["tms_prompt_multi_arm"] = """
从TMS研究中提取参数，返回JSON：

{content}

{{
    "intervention_groups": [
        {{
            "group_id": 1,
            "group_name": "组别名称",
            "tms_type": "TMS类型",
            "stimulation_frequency": 频率(Hz),
            "stimulation_intensity": "强度(%MT)",
            "session_number": 治疗次数,
            "brain_target": "刺激靶点",
            "hemisphere": "半球(left/right)",
            "total_number_pulses_per_session": "每个会话的脉冲总数",
            "train_duration": "单串刺激持续时间",
            "train_time_interval": "训练时间间隔",
            "inter_session_interval": "会话时间间隔",
        }},
        {{
            "group_id": 2,
            "group_name": "组别名称",
            "tms_type": "TMS类型",
            "stimulation_frequency": 频率(Hz),
            "stimulation_intensity": "强度(%MT)",
            "session_number": 治疗次数,
            "brain_target": "刺激靶点",
            "hemisphere": "半球(left/right)",
            "total_number_pulses_per_session": "每个会话的脉冲总数",
            "train_duration": "单串刺激持续时间",
            "train_time_interval": "训练时间间隔",
            "inter_session_interval": "会话时间间隔",
        }},
        ...
    ],
    "control_group": {{
        "group_id": 3,
        "group_name": "组别名称",
        "coil_tilt_angle_for_sham": "假刺激时线圈倾斜角度",
        "sham_type": "假刺激方式"
    }}
}}

注意：
1.频率采用字符串形式，如果刺激是按簇进行，则仿照"50Hz bursts at 5Hz"生成
2.强度如果是resting motor threshold，则记为"xx% of RMT"，如果是active motor threshold，则记为"xx% of AMT"
3.会话时间间隔是用于记录accelerated TMS中两次会话的间隔，如果不是aTMS或者间隔时间大于1天，则记为null
4.如果分组数量大于3，则在省略号后进行拓展，否则保持完整的json格式

只返回JSON，无信息设null。
"""

PROMPTS["tms_prompt"] = """
从TMS研究中提取参数，返回JSON：

{content}

{{
    "tms_type": "TMS类型",
    "stimulation_frequency": 频率(Hz),
    "stimulation_intensity": "强度(%MT)",
    "session_number": 治疗次数,
    "brain_target": "刺激靶点",
    "hemisphere": "半球(left/right)",
    "total_number_pulses_per_session": "每个会话的脉冲总数",
    "train_duration": "单串刺激持续时间",
    "train_time_interval": "训练时间间隔",
    "inter_session_interval": "会话时间间隔",
    "coil_tilt_angle_for_sham": "假刺激时线圈倾斜角度",
    "sham_type": "假刺激方式"
}}

注意：
1.频率采用字符串形式，如果刺激是按簇进行，则仿照"50Hz bursts at 5Hz"生成
2.强度如果是resting motor threshold，则记为"xx% of RMT"，如果是active motor threshold，则记为"xx% of AMT"
3.会话时间间隔是用于记录accelerated TMS中两次会话的间隔，如果不是aTMS或者间隔时间大于1天，则记为null

只返回JSON，无信息设null。
"""

PROMPTS["rob2_prompt"]="""
根据RoB2工具评估以下RCT研究的偏倚风险，返回JSON格式：

{combined_content}

请按照RoB2的5个领域进行评估：

{{
    "domain1_randomization": {{
        "risk_level": "Low/Some concerns/High",
        "rationale": "评估理由",
        "supporting_evidence": "支持证据"
    }},
    "domain2_deviations": {{
        "risk_level": "Low/Some concerns/High", 
        "rationale": "评估理由",
        "supporting_evidence": "支持证据"
    }},
    "domain3_missing_data": {{
        "risk_level": "Low/Some concerns/High",
        "rationale": "评估理由", 
        "supporting_evidence": "支持证据"
    }},
    "domain4_outcome_measurement": {{
        "risk_level": "Low/Some concerns/High",
        "rationale": "评估理由",
        "supporting_evidence": "支持证据"
    }},
    "domain5_selective_reporting": {{
        "risk_level": "Low/Some concerns/High",
        "rationale": "评估理由",
        "supporting_evidence": "支持证据"
    }},
    "overall_bias_risk": {{
        "risk_level": "Low/Some concerns/High",
        "rationale": "总体评估理由"
    }}
}}

RoB2评估标准：
1. Domain 1 (随机化过程): 随机序列生成、分配隐藏、基线平衡
2. Domain 2 (偏离预期干预): 盲法、依从性、意向性治疗分析
3. Domain 3 (缺失结局数据): 脱落率、缺失数据处理、敏感性分析
4. Domain 4 (结局测量): 测量者盲法、测量工具适当性
5. Domain 5 (选择性报告): 预先注册、所有预设结局均报告

注意：
在进行总体评估时，由于木桶原理，整体偏倚风险应该为5个Domain中的最高风险等级。

评估等级：
- Low: 低偏倚风险
- Some concerns: 存在一些担忧
- High: 高偏倚风险

只返回JSON格式，基于文中实际信息进行评估。
"""

PROMPTS["summary_prompt"] = """
请基于以下论文关键部分，用英文生成一个结构化的全文摘要，重点关注：

1. 研究设计和方法
2. 被试特征和干预措施
3. 主要结果和统计数据
4. 结论和临床意义

论文内容：
{chunks_text}

要求：
1.摘要长度在1000个英文单词左右
2.如果所给论文内容不包括重点关注的部分，则不需要在摘要中提及
3.仅生成所需摘要，不要给出多余的建议

请生成简洁但全面的摘要：
"""

PROMPTS["keyword_prompt"] = """
从以下{text_type}中提取关键词，并按医学研究PICO框架分类：

1. Population（人群）：研究对象、疾病类型、年龄群体
2. Intervention（干预）：治疗方法、刺激参数、设备
3. Outcome（结局）：评估工具、测量指标
4. Design（设计）：试验设计、盲法、随机化
5. General（一般）：其他重要关键词

文本内容：
{text}

要求：
1.关键词为英文
2.对于干预部分的关键词提取要准确，比如文中明确说accelerated TMS、deep TMS，这些前缀需要完整的记录
3.对于干预部分，必须记录所用的刺激方法，比如”rTMS“,"TMS"不能因为刺激方式的常规而不记录
3.对于干预部分，如果出现TMS联合某种治疗方式，需要将该治疗方式记录下来
4.对于干预部分，如果刺激频率小于等于1Hz，则记录”low frequency“，如果大于等于5Hz，则记录”high frequency“
5.对于干预部分，必须要将刺激类型记录，不能仅记录刺激参数等细节信息
6.如果找不到相关关键词，返回空列表

请返回JSON格式，每类3-8个关键词：
{{
    "population": ["关键词1", "关键词2"],
    "intervention": ["关键词1", "关键词2"],
    "outcome": ["关键词1", "关键词2"],
    "design": ["关键词1", "关键词2"],
    "general": ["关键词1", "关键词2"]
}}
"""


PROMPTS["query_prompt"] = """
从以下查询中提取关键词和排除关键词，并按医学研究PICO框架分类，提取结果用英文表示，重点关注：

1. Population（人群）：研究对象、疾病类型、年龄群体
2. Intervention（干预）：治疗方法、刺激参数、设备
3. Outcome（结局）：评估工具、测量指标
4. Design（设计）：试验设计、盲法、随机化
5. General（一般）：其他重要关键词
6. Must-Have（必须）：必须出现的关键词（如联合疗法、特定技术）
7. Exclude（排除）：用户明确要排除的内容
8. Year Range（年份范围）：研究发表的年份限制

查询：{query}

请按以下JSON格式返回：
{{
    "include_keywords": {{
        "population": ["关键词1", "关键词2"],
        "intervention": ["关键词1", "关键词2"],
        "outcome": ["关键词1", "关键词2"],
        "design": ["关键词1", "关键词2"],
        "general": ["关键词1", "关键词2"]
    }},
    "must_have_keywords": ["必须词1", "必须词2", ...],
    "exclude_keywords": ["排除词1", "排除词2", ...],
    "year_range": {{
        "start": 起始年份或null,
        "end": 结束年份或null
    }}
}}

要求：
1. 关键词必须为英文
2. 对于干预部分的关键词提取要准确，比如文中明确说accelerated TMS、deep TMS，这些前缀需要完整的记录
3. 如果查询没有提到某些分类，该分类返回空列表
4. **识别必须匹配的关键词（must_have_keywords）**：
   - 联合疗法：如 "TMS + cognitive training" → must_have: ["cognitive training"]，注意仅记录“+”后面的关键词，不要进行扩写
   - 特定技术组合：如 "rTMS combined with rehabilitation" → must_have: ["rehabilitation"]，注意仅记录“combined with”后面的关键词，不要进行扩写。
   - 查询中明确提到必要的关键词：如“the necessary keyword”等
5. 仔细识别用户想要排除的内容，常见的排除表达包括：
   - 英文："exclude", "excluding", "without", "not including", "except", "not", "no"等
   - 中文："排除", "不包括", "除外", "不要", "去除", "不含"等
6. 如果没有排除关键词，exclude_keywords 返回空数组 []
7. 排除关键词要扩展同义词，例如 "TRD" 应该包括 "treatment-resistant depression", "refractory depression"
8. 识别年份范围的常见表达：
   - "after 2015", "since 2015", "from 2015" → {{"start": 2015, "end": null}}
   - "before 2020", "until 2020", "up to 2020" → {{"start": null, "end": 2020}}
   - "between 2015 and 2020", "from 2015 to 2020", "2015-2020" → {{"start": 2015, "end": 2020}}
   - "in 2020", "published in 2020" → {{"start": 2020, "end": 2020}}
   - "recent 5 years", "last 5 years" → 计算为当前年份-5到当前年份
   - 中文："2015年之后", "2015年以来", "2015-2020年", "最近5年"
9. 如果没有提到年份，year_range 返回 {{"start": null, "end": null}}
10. 如果用户询问的疾病不是具体的疾病名称，比如“neurodegeneration”就需要将最常见的"Alzheimer's disease","Parkinson"等神经退行性疾病记录
11. 只返回JSON格式，不要其他解释

示例：
查询：How effective is rTMS the necessary keyword for depression after 2015, excluding treatment-resistant depression?
返回：
{{
    "include_keywords": {{
        "population": ["depression", "depressive disorder", "major depressive disorder"],
        "intervention": ["rtms", "repetitive transcranial magnetic stimulation"],
        "outcome": ["efficacy", "effectiveness", "treatment response"],
        "design": [],
        "general": []
    }},
    "must_have_keywords": ["cognitive training", "cognitive therapy", "cognitive rehabilitation"],
    "exclude_keywords": ["treatment-resistant depression", "trd", "refractory depression", "treatment-refractory depression"],
    "year_range": {{"start": 2015, "end": null}}
}}

"""

PROMPTS["timepoint_prompt"] = """
你正在辅助进行Meta分析。一篇论文中有多个时间点的数据，请根据用户的研究问题和时间点选择原则，选出最适合的效应量数据。

用户的研究问题：
{query_context}

可用的时间点数据：
{timepoint_descriptions}

选择原则（按优先级排序）：
1. 排除所有baseline/治疗前的时间点
2. 根据用户研究问题选择最相关的时间点：
   - 如果用户关注的是"短期效果/即时效果"，优先选择治疗刚结束或最早的治疗后时间点
   - 如果用户关注的是"长期效果/持续效果/随访效果"，优先选择最长的随访时间点
   - 如果用户没有明确偏好，默认选择第一个治疗后的时间点（post-treatment）
3. 如果时间点信息不足以判断先后顺序，选择数据质量更好的那个

请只返回最佳选项的索引数字（0, 1, 2等），不要其他解释。
"""

# 当无法提供回答时，返回的默认提示语句。
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

# 加载动画
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# 默认文本分隔符，用于文本分隔。
PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
