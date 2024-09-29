"""LLM Compiler Prompts.

Taken from https://github.com/SqueezeAILab/LLMCompiler.

"""

# ruff: noqa: E501

JOINER_FINISH = "finish"
END_OF_PLAN = "<END_OF_PLAN>"


# This is a mashup of different prompts in https://github.com/SqueezeAILab/LLMCompiler/blob/main/configs.
PLANNER_EXAMPLE_PROMPT = (
    "Question: If cheetah was 1.3 times slower, greyhound was 1.5 times faster, and falcon was 2.3 time slower then their maximum speeds, "
    "what will be the ratio of the fastest animal to the slowest animal?\n"
    '1. search("cheetah")\n'
    '2. math("cheetah max speed in km/h if 1.3 times slower?", ["$1"]\n'
    '3. search("greyhound")\n'
    '4. math("greyhound max speed in km/h if 1.5 times faster?", ["$3"]\n'
    '5. search("falcon")\n'
    '6. math("falcon max speed in km/h if 2.3 times slower?", ["$5"]\n'
    '7. math("max($2, $4, $6) / min($2, $4, $6)")\n'
    "Thought: I can answer the question now.\n"
    f"8. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Find a movie similar to Mission Impossible, The Silence of the Lambs, American Beauty, Star Wars Episode IV - A New Hope\n"
    "Options:\n"
    "Austin Powers International Man of Mystery\n"
    "Alesha Popvich and Tugarin the Dragon\n"
    "In Cold Blood\n"
    "Rosetta\n"
    "Thought: I need to find all movies in the Question.\n"
    '1. search("Mission Impossible")\n'
    '2. search("The Silence of the Lambs")\n'
    '3. search("American Beauty")\n'
    '4. search("Star Wars Episode IV - A New Hope")\n'
    "Thought: I need to find all movies in the Options.\n"
    "5. search(Austin Powers International Man of Mystery)\n"
    "6. search(Alesha Popvich and Tugarin the Dragon)\n"
    "7. search(In Cold Blood)\n"
    "8. search(Rosetta)\n"
    "Thought: I can answer the question now.\n"
    f"9. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
    '1. search("Arthur\'s Magazine")\n'
    '2. search("First for Women (magazine)")\n'
    "Thought: I can answer the question now.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n"
    '1. search("Pavel Urysohn")\n'
    '2. search("Leonid Levin")\n'
    "Thought: I can answer the question now.\n"
    f"3. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: Determine the smaller value: the depth difference in meters between the Mariana Trench and the Puerto Rico Trench, "
    "or the depth difference in meters between the South Sandwich Trench and the Sunda Trench.\n"
    "1. search('Mariana Trench')\n"
    "2. search('Puerto Rico Trench')\n"
    "3. math('absolute depth difference between Mariana and Puerto Rico Trench in meters?', ['$1', '$2'])\n"
    "4. search('South Sandwich Trench')\n"
    "5. search('Sunda Trench')\n"
    "6. math('absolute depth difference between South Sandwich and Sunda Trench in meters?', ['$4', '$5'])\n"
    "7. math('min($3, $6)')\n"
    "Thought: I can answer the question now.\n"
    f"8. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: What is the raio of the height of Mount Everest and the height of Mount Kilimanjaro?\n"
    "1. search('Mount Everest')\n"
    "2. search('Mount Kilimanjaro')\n"
    "3. math('height of Mount Everest / height of Mount Kilimanjaro', ['$1', '$2'])\n"
    "Thought: I can answer the question now.\n"
    f"4. join(){END_OF_PLAN}\n"
    "###\n"
)

OUTPUT_PROMPT = (
    "Solve a question answering task with interleaving Observation, Thought, and Action steps. "
    # "You will be given a question and some Wikipedia passages, which are the observations.\n\n"
    "Thought step can reason about the observations in 1-2 sentences.\n"
    "Action can be only one type:"
    f" (1) {JOINER_FINISH}(answer): returns the answer and finishes the task. "
    "    - Answer can be the thought directly, or a single number, or a single string if that's best for the question.\n"
    "    - Answer can be long or short, but it should be a single answer.\n"
    "For instance, when you are asked about the ratio of the height of Mount Everest and the height of Mount Kilimanjaro, you must "
    "return the ratio (e.g. 1.2), not the name of the mountain (e.g. Mount Everest to Mount Kilimanjaro).\n"
    "\n"
    "Here are some examples:\n"
    "\n"
    "Question: If cheetah was 1.3 times slower, and greyhound was 1.5 times faster, which animal was faster?\n"
    "search(cheetah)\n"
    "Observation: The cheetah (Acinonyx jubatus) is a large cat with a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. Its head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94\xa0cm (26–37\xa0in) at the shoulder, and the head-and-body length is between 1.1 and 1.5\xa0m (3\xa0ft 7\xa0in and 4\xa0ft 11\xa0in). Adults weigh between 21 and 72\xa0kg (46 and 159\xa0lb). It is the fastest land animal, capable of running at 80 to 98\xa0km/h (50 to 61\xa0mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail..\n"
    "search(greyhound)\n"
    'Observation: The English Greyhound, or simply the Greyhound, is a breed of dog, a sighthound which has been bred for coursing, greyhound racing and hunting. Since the rise in large-scale adoption of retired racing Greyhounds, the breed has seen a resurgence in popularity as a family pet.. Greyhounds are defined as a tall, muscular, smooth-coated, "S-shaped" type of sighthound with a long tail and tough feet. Greyhounds are a separate breed from other related sighthounds, such as the Italian greyhound.[2][3]. The Greyhound is a gentle and intelligent breed whose combination of long, powerful legs, deep chest, flexible spine, and slim build allows it to reach average race speeds exceeding 64 kilometres per hour (40\xa0mph).[4][5][6] The Greyhound can reach a full speed of 70 kilometres per hour (43\xa0mph) within 30 metres (98\xa0ft), or six strides from the boxes, traveling at almost 20 metres per second (66\xa0ft/s) for the first 250 metres (820\xa0ft) of a race.[7][8].\n'
    "math(what is the speed of cheetah in km/h if it was 1.3 times slower?)\n"
    "Observation: 61.53846153846154\n"
    "math(what is the speed of greyhound in km/h if it was 1.5 times faster?)\n"
    "Observation: 105\n"
    "Thought: Comparing the numbers, the greyhound was faster.\n"
    f"Action: {JOINER_FINISH}(greyhound)\n"
    "###\n"
    "\n"
    "Question: If Mount Everest's height were halved and Mount Kilimanjaro's height were doubled, what would be the difference in their height?\n"
    "search(Mount Everest)\n"
    'Observation: Mount Everest (Nepali: सगरमाथा, romanized:\xa0Sagarmāthā; Tibetan: Chomolungma ཇོ་མོ་གླང་མ; Chinese: 珠穆朗玛峰; pinyin: Zhūmùlǎngmǎ Fēng) is Earth\'s highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point.[2] Its elevation (snow height) of 8,848.86\xa0m (29,031\xa0ft 82\xa0in) was most recently established in 2020 by the Chinese and Nepali authorities.[3][4]. Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the "standard route") and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall.\n'
    "search(Mount Kilimanjaro)\n"
    "Observation: Mount Kilimanjaro (/ˌkɪlɪmənˈdʒɑːroʊ/)[4] is a dormant volcano located in Kilimanjaro Region of Tanzania. It has three volcanic cones: Kibo, Mawenzi, and Shira. It is the highest mountain in Africa and the highest single free-standing mountain above sea level in the world: 5,895\xa0m (19,341\xa0ft) above sea level and about 4,900\xa0m (16,100\xa0ft) above its plateau base. It is the highest volcano in Africa and the Eastern Hemisphere.. Kilimanjaro is the fourth most topographically prominent peak on Earth.\n"
    "math(what is the half of the height of Mount Everest in meter?)\n"
    "Observation: 4444.43\n"
    "math(what is the double of the height of Mount Kilimanjaro in meter?)\n"
    "Observation: 11790\n"
    "math(abs(4444.43 - 11790))\n"
    "Observation: 7345.57\n"
    "Thought: Difference in their height is 7345.57.\n"
    f"Action: {JOINER_FINISH}(7345.57)\n"
    "###\n"
    "\n"
    "Question: With the Sahara Desert's area reduced by 33% and the Kalahari Desert's area magnified by 52%, which one covers more ground?\n"
    "search(Sahara Desert)\n"
    'Observation: The Sahara (/səˈhɑːrə/, /səˈhærə/) is a desert spanning North Africa. With an area of 9,200,000 square kilometres (3,600,000\xa0sq\xa0mi), it is the largest hot desert in the world and the third-largest desert overall, smaller only than the deserts of Antarctica and the northern Arctic.[1][2][3]. The name "Sahara" is derived from the Arabic word for "desert" in the feminine irregular form, the singular ṣaḥra\' (صحراء /ˈsˤaħra/), plural ṣaḥārā (صَحَارَى /ˈsˤaħaːraː/),[4][5][6][7] ṣaḥār (صَحَار), ṣaḥrāwāt (صَحْرَاوَات), ṣaḥār).. The desert covers much of North Africa, excluding the fertile region on the Mediterranean Sea coast, the Atlas Mountains of the Maghreb, and the Nile Valley in Egypt and the Sudan.[8]. It stretches from the Red Sea in the east and the Mediterranean in the north to the Atlantic Ocean in the west, where the landscape gradually changes from desert to coastal plains.'
    "search(Kalahari Desert)\n"
    'Observation: The Kalahari Desert is a large semi-arid sandy savanna in Southern Africa extending for 900,000 square kilometres (350,000\xa0sq\xa0mi), covering much of Botswana, as well as parts of Namibia and South Africa.. It is not to be confused with the Angolan, Namibian, and South African Namib coastal desert, whose name is of Khoekhoegowab origin and means "vast place".. Kalahari is derived from the Tswana word Kgala, meaning "the great thirst", or Kgalagadi, meaning "a waterless place";[1] the Kalahari has vast areas covered by red sand without any permanent surface water.. The Kalahari Desert was not always a dry desert. The fossil flora and fauna from Gcwihaba Cave in Botswana indicates that the region was much wetter and cooler at least from 30 to 11 thousand BP (before present), especially after 17,500 BP.[2].'
    "math(what the area of the Sahara Desert in km^2 if it was reduced by 33%?)\n"
    "Observation: 6164000\n"
    "math(what the area of the Kalahari Desert in km^2 if it was magnified by 52%?)\n"
    "Observation: 1368000\n"
    "Thought: Comparing the numbers, the Sahara Desert covers more ground.\n"
    f"Action: {JOINER_FINISH}(Sahara Desert)\n"
    "###\n"
    "\n"
    "Question: Determine the smaller value: the depth difference in meters between the Mariana Trench and the Puerto Rico Trench, or the depth difference in meters between the South Sandwich Trench and the Sunda Trench.\n"
    "search(Mariana Trench)\n"
    "Observation: The Mariana Trench is an oceanic trench located in the western Pacific Ocean, about 200 kilometres (124\xa0mi) east of the Mariana Islands; it is the deepest oceanic trench on Earth. It is crescent-shaped and measures about 2,550\xa0km (1,580\xa0mi) in length and 69\xa0km (43\xa0mi) in width. The maximum known depth is 10,984\xa0±\xa025 metres (36,037\xa0±\xa082\xa0ft; 6,006\xa0±\xa014 fathoms; 6.825\xa0±\xa00.016\xa0mi) at the southern end of a small slot-shaped valley in its floor known as the Challenger Deep.[1] The deepest point of the trench is more than 2\xa0km (1.2\xa0mi) farther from sea level than the peak of Mount Everest.[a]. At the bottom of the trench, the water column above exerts a pressure of 1,086\xa0bar (15,750\xa0psi), more than 1,071 times the standard atmospheric pressure at sea level. At this pressure, the density of water is increased by 4.96%.\n"
    "search(Puerto Rico Trench)\n"
    "Observation: The Puerto Rico Trench is located on the boundary between the Caribbean Sea and the Atlantic Ocean. The oceanic trench, the deepest in the Atlantic, is associated with a complex transition between the Lesser Antilles subduction zone to the south and the major transform fault zone or plate boundary, which extends west between Cuba and Hispaniola through the Cayman through to the coast of Central America.. The trench is 800 kilometres (497\xa0mi) long[1] and has a maximum depth of 8,376 metres (27,480\xa0ft)[2] or 5.20 miles. This constitutes the single deepest point in the Atlantic Ocean. This point is commonly referred to as the Milwaukee Deep, with the Brownson Deep naming the seabed surrounding it.[3] However, more recently, the latter term has also been used interchangeably with the former to refer to this point.[4][5][6] The exact point was identified by the DSSV Pressure Drop using a state-of-the-art Kongsberg EM124 multibeam sonar in 2018, and then directly visited and its depth verified by the crewed submersible Deep-Submergence Vehicle DSV Limiting Factor (a Triton 36000/2 model submersible) piloted by Victor Vescovo.[7][8][9]."
    "math(what is the depth difference between the Mariana Trench and the Puerto Rico Trench in meters?)\n"
    "Observation: 2608\n"
    "search(South Sandwich Trench)\n"
    "Observation: 55°25′44″S 26°11′29″W\ufeff / \ufeff55.42889°S 26.19139°W\ufeff / -55.42889; -26.19139. The South Sandwich Trench is a deep arcuate trench in the South Atlantic Ocean lying 100 kilometres (62\xa0mi) to the east of the South Sandwich Islands. It is the deepest trench of the Southern Atlantic Ocean, and the second deepest of the Atlantic Ocean after the Puerto Rico Trench. Since the trench extends south of the 60th parallel south, it also contains the deepest point in the Southern Ocean.. The deepest point in the entire trench is the Meteor Deep, whose location prior to February 2019 was identified as 55°25.12′S 26°24.28′W\ufeff / \ufeff55.41867°S 26.40467°W\ufeff / -55.41867; -26.40467\ufeff (Meteor Deep) at a depth of 8,202 metres (26,909\xa0ft).\n"
    "search(Sunda Trench)\n"
    "Observation: The Sunda Trench, earlier known as and sometimes still indicated as the Java Trench,[1] is an oceanic trench located in the Indian Ocean near Sumatra, formed where the Australian-Capricorn plates subduct under a part of the Eurasian Plate. It is 3,200 kilometres (2,000\xa0mi) long with a maximum depth of 7,290 metres (23,920 feet).[2] Its maximum depth is the deepest point in the Indian Ocean. The trench stretches from the Lesser Sunda Islands past Java, around the southern coast of Sumatra on to the Andaman Islands, and forms the boundary between Indo-Australian Plate and Eurasian plate (more specifically, Sunda Plate). The trench is considered to be part of the Pacific Ring of Fire as well as one of a ring of oceanic trenches around the northern edges of the Australian Plate.. In 2005, scientists found evidence that the 2004 earthquake activity in the area of the Java Trench could lead to further catastrophic shifting within a relatively short period of time, perhaps less than a decade.[3] This threat has resulted in international agreements to establish a tsunami warning system in place along the Indian Ocean coast.[4]."
    "math(what is the depth difference between the South Sandwich Trench and the Sunda Trench in meters?)\n"
    "Observation: 912\n"
    "math(min(2608, 912))\n"
    "Observation: 912\n"
    "Thought: The smaller depth difference is 912\n"
    f"Action: {JOINER_FINISH}(912)\n"
    "###\n"
    "\n"
    "Question: {query_str}\n\n"
    "{context_str}\n"
)


"""Default prompt for ReAct agent."""
from pathlib import Path

# TODO: have formatting instructions be a part of react output parser
with (
    Path(__file__).parents[0] / Path("templates") / Path("system_header_template.md")
).open("r") as f:
    __BASE_REACT_CHAT_SYSTEM_HEADER = f.read()

REACT_CHAT_SYSTEM_HEADER = __BASE_REACT_CHAT_SYSTEM_HEADER.replace(
    "{context_prompt}", "", 1
)

CONTEXT_REACT_CHAT_SYSTEM_HEADER = __BASE_REACT_CHAT_SYSTEM_HEADER.replace(
    "{context_prompt}",
    """
Here is some context to help you answer the question and plan:
{context}
""",
    1,
)


STEP_EXAMPLE = {
    "obversation_feedback": "observation",
    "relevance": "True",
    "new_search_actions": [
        {"action": "search_func", "action_input": {"func_name": "str"}},
        {
            "action": "search_method_in_class",
            "action_input": {"class_name": "str", "method_name": "str"},
        },
    ],
}

BUG_OUTPUT = {
    "bug_locations": [
        {
            "file": "path/to/file",
            "class": "class_name",
            "method": "function_name",
        },
        {
            "file": "path/to/file",
            "class": "class_name",
            "method": "function_name",
        },
    ]
}

SEARCH_SYSTEM_HEADER = r"""
You are a helpful assistant that use API calls to report bug code snippets from a text into json format.
You need to extract where are the bug locations by analyzing the text.
The given text will give you suspicous_code containing keyword. Make sure to search for the keyword in the codebase.
There are some API calls that you can use to extract the information.
The API calls include:
{tool_desc}

You also need to keep in ind the priority of these API calls.
{priority_desc}

Everytime you will do the following things:

Provide the observation based on given input. Check whether it contains any class, method or function you need to further search.
Notice that you should put new classes, methods or functions related to your current code snippets. Don't put any class, method or function we currently haven't searched.
You can put multiple classes, methods or functions in the new_search_actions list.
If you make sure the context is enough to answer the question, you can keep the new_search_actions list empty.

Conclusion is a final standalone step to provide the final bug locations when nothing else to search.

## Output format
1. Regular Step Format:
    Provide your answer in a clear JSON structure like this,
    {step_format}
    Make sure each API call is written as a valid python expression and code_snippet is a valid python string.
    In relevance, set to True if the given context is relevant to the bug location, otherwise set to False.
    You can provide multiple actions in the new_search_actions. DO NOT add any title or description.
2. Conclusion Format:
    After confirming you have enough context to answer the question, provide the final bug locations in JSON structure like this,
    DO NOT mix it with the observation or add any title or description. If method is not belong to any class, set class to empty string.
    {bug_locations}

"""

EXTRACT_FORMATS = {
    "slice": {
        "traceback_warning_log_slice": "log_slice_string",
        "issue_reproducer_slice": "code_slice_string",
        "source_code_slice": "code_slice_string",
    },
    "parse": {
        "code_info_list": [
            {"keyword": "class_or_function_name_1", "file_path": ""},
            {"keyword": "class_or_function_name_2", "file_path": "file_path_2"},
        ]
    },
    "judge": {"is_successful": True},
    "summarize": {
        "summary": "summary_string",
        "code_info_list": [
            {"keyword": "class_or_function_name_1", "file_path": ""},
            {"keyword": "class_or_function_name_2", "file_path": "file_path_2"},
        ],
    },
}

EXTRACT_FIELDS = {
    "slice": """
<field>
    traceback_warning_log_slice: Traceback or warning log. Set to '' if not found.
</field>
<field>
    issue_reproducer_slice: Code snippet to reproduce the issue. Should be a python code snippet that can be directly runned.
            \n should be used for new line, 4 spaces should be used for indentation.
            If file creation is necessary, python file IO should be used.
            If the reproducer is mentioned in interactive mode, the code should be extracted and parsed into an .py file.
            For example, '>>> ' should never be used in an .py file, and the output of interactive shell should also be commented.
            Code shouldn't be inferred from natural language description. Set to '' if not found.
</field>
<field>
    source_code_slice: Code referenced in the issue which comes from the source repo. Should have python code only.
            DO NOT label code as this category UNLESS EXPLICITE words are found,
            saying that it is COPIED / REFERENCED from the source repo.
            Shouldn't overlap with traceback_warning_log_slice or issue_reproducer_slice.
            Set to '' if no code satisfies this requirement.
</field>
""",
    "parse": """
<field>
    keyword: the name of the class or function where the suspicious code lies in.
            Should be a single word, not spliced with dot.
</field>
<field>
    file_path: The path of the file containing the code. Can be relative or absolute path.
            Levels of path should only be spliced with slash or backslash, not space.
            Specially, python import style path should be parsed as:
            1. dot replaced with slash;
            2. add .py suffix if no suffix is found.
            For example, "pvlib.bifacial.pvfactors" should be interpreted as "pvlib/bifacial/pvfactors.py"
            Set to '' if cannot find path.
</field>
<field>
    code_info_list: list of (keyword, file_path). All keywords mentioned should be extracted to the list.
</field>
""",
    "judge": """
<field>
    is_successful: whether the reproduce_snippet successfully reproduced the issue, based on the reproducer_log it generated.
            Note that 'successfully reproduce' means the similar phenomenon is observed;
            It does not necessarily means the snippet finished without error
            (Getting the same error reported in issue means reproduction is successful)
</field>
""",
    "summarize": """
<field>
    summary: Summary in natural language. Requirements include:
            1. Describe the issue;
            2. Suggest the methods/classes/functions/files that following agents should examine;
            3. Be within 50 words.
</field>
<field>
    code_info_list: list of (keyword, file_path).
            All keywords mentioned in natural language (not code snippet or traceback) should be extracted to the list.
</field>
<field>
    keyword: the name of the class or function where the suspicious code lies in.
            Should be a single word, not spliced with dot.
</field>
<field>
    file_path: The path of the file containing the code. Can be relative or absolute path.
            Levels of path should only be spliced with slash or backslash, not space.
            Specially, python import style path should be parsed as:
            1. dot replaced with slash;
            2. add .py suffix if no suffix is found.
            For example, "pvlib.bifacial.pvfactors" should be interpreted as "pvlib/bifacial/pvfactors.py"
            Set to '' if cannot find path.
</field>
""",
}

EXTRACT_EXAMPLES = {
    "slice": {
        "repo_name": "marshmallow-code/marshmallow",
        "input_description": """
3.0: DateTime fields cannot be used as inner field for List or Tuple fields

`DateTime` fields have started throwing an error when being instantiated as inner fields of container fields like `List` or `Tuple`.

```python
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

s = MySchema()
```

Traceback:
```
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    s = MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
AttributeError: 'List' object has no attribute 'opts'
```

It seems like it's treating the parent field as a Schema without checking that it is indeed a schema.
""",
        "example_output": {
            "traceback_warning_log_slice": """
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    s = MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
AttributeError: 'List' object has no attribute 'opts'
""",
            "issue_reproducer_slice": """
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

s = MySchema()
""",
            "source_code_slice": "",
        },
    },
    "parse": {
        "traceback": {
            "repo_name": "marshmallow-code/marshmallow",
            "input_description": """
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    s = MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py", line 636, in _bind_to_schema
    self.inner._bind_to_schema(field_name, self)
AttributeError: 'List' object has no attribute 'opts'
""",
            "example_output": {
                "code_info_list": [
                    {"keyword": "<module>", "file_path": "test-mm.py"},
                    {
                        "keyword": "__init__",
                        "file_path": "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py",
                    },
                    {
                        "keyword": "_bind_to_schema",
                        "file_path": "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py",
                    },
                ]
            },
        },
        "code": {
            "repo_name": "marshmallow-code/marshmallow",
            "input_description": """
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

s = MySchema()
""",
            "example_output": {
                "code_info_list": [
                    {"keyword": "fields", "file_path": ""},
                    {"keyword": "Schema", "file_path": ""},
                ]
            },
        },
    },
    "summarize": {
        "repo_name": "marshmallow-code/marshmallow",
        "input_description": """
3.0: DateTime fields cannot be used as inner field for List or Tuple fields

`DateTime` fields have started throwing an error when being instantiated as inner fields of container fields like `List` or `Tuple`.

```python
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

s = MySchema()
```

Traceback:
```
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    s = MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
AttributeError: 'List' object has no attribute 'opts'
```

It seems like it's treating the parent field as a Schema without checking that it is indeed a schema.
""",
        "example_output": {
            "summarize": """
In marshmallow 3.0, using DateTime fields as inner fields in List or Tuple containers triggers an AttributeError.
The error occurs because List is mistakenly treated as a schema.
Examine the fields.List, fields.DateTime, and _init_fields methods in schema.py for debugging.
""",
            "code_info_list": [
                {"keyword": "DateTime", "file_path": ""},
                {"keyword": "Schema", "file_path": ""},
                {"keyword": "opts", "file_path": ""},
            ],
        },
    },
}

EXTRACT_PROMPTS = {
    "header": """
You are an expert python developer, mastering at summarizing and extracting from github issues.
""",
    "example": r"""
<repo_name>{example_repo_name}<repo_name>
<example_input_description>
{example_input_description}
</example_input_description>
<example_output>
{example_output}
</example_output>
""",
    "slice": r"""
Your task is to slice strings from human reported github issue.
Every slice shouldn't overlap with another slice. Non-existanct slice should be set to ''.

Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)

The meanings of each field are:
{output_fields}

An example is given below:
{example}

Below is the real task for you to solve:
<repo_name>{repo_name}</repo_name>
{input_description}
""",
    "parse": r"""
Your task is to extract python code keywords and the filepath they belong to (if exist) from human reported github issue.
Non-existanct filepath should be set to ''.

Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)

The meanings of each field are:
{output_fields}

An example is given below:
{example}

Below is the real task for you to solve:
<repo_name>{repo_name}</repo_name>
<input_description>
{input_description}
</input_description>
""",
    "judge": r"""
Your task is to judge whether an input github issue is successfully reproduced by a reproduce_snippet based on the reproducer_log it generated.
Non-existanct filepath should be set to ''.

Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)

The meanings of each field are:
{output_fields}

Below is the real task for you to solve:
<repo_name>{repo_name}</repo_name>
<input_description>
{input_description}
</input_description>
<reproduce_snippet>
{reproduce_snippet}
</reproduce_snippet>
<reproducer_log>
{reproducer_log}
</reproducer_log>
""",
    "summarize": r"""
Your task is to summarize a human reported github issue in natural language.

Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)

The meanings of each field are:
{output_fields}

An example is given below:
{example}

Below is the issue for you to summarize:
<repo_name>{repo_name}</repo_name>
<input_description>
{input_description}
</input_description>
""",
}
