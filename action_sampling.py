"""
This module extracts instructions using a specified model from OpenAI.
"""
import sys
import os
import argparse
from contextlib import redirect_stdout
import instructor
import math
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from time import time

from instructor import Mode
from typing import Callable, List, Optional, Tuple, Type

# see: load environment variables from .env file
load_dotenv()

class UserInfo(BaseModel):
    keywords: Optional[List[str]]
    steps: Optional[List[str]]
    
class Tee:
	def __init__(self, *file_objects):
		self.file_objects = file_objects
	
	def write(self, text):
		for file_object in self.file_objects:
			file_object.write(text)
	
	def flush(self):
		for file_object in self.file_objects:
			file_object.flush()

def conf():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--openai_api_base', type=str, default="https://api.openai.com/v1", help='Model Server URL.')
    # parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Model to use for instruction extraction.')
    parser.add_argument('--openai_api_base', type=str, default="http://nlp-in-477-l:8000/v1", help='Model Server URL.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Model to use for instruction extraction.')
    # parser.add_argument('--model', type=str, default="McGill-NLP/Llama-3-8B-Web", help='Model to use for instruction extraction.')
    # parser.add_argument('--model', type=str, default="allenai/Llama-3.1-Tulu-3-8B-SFT", help='Model to use for instruction extraction.')
    # parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='Model to use for instruction extraction.')
    return parser.parse_args()

if __name__ == "__main__":
    args = conf()
    openai_api_base, model = args.openai_api_base, args.model
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = os.environ.get(f"{model}:api_key")
    openai_api_base = os.environ.get(f"{model}:url_base")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    # see: 128k context of llama3.1
    # completion = client.completions.create(model=model, prompt="San Francisco is a")
    payload = [
        {
            "role": "user",
            "content": "You are an assistant to help users extract the most informative instructions from the input."
        },
        {
            "role": "user",
            "content":  """You are a UI Assistant. Your goal is to help the user perform tasks using a web browser. You communicate with the user via chat, where the user provides instructions and you can send messages back. You have access to a web browser that both you and the user can see, and only you can interact with via specific commands. Review the user's instructions, the current state of the web page, and the history of interactions to determine the best next action to accomplish your goal. Your response will be interpreted and executed by a program; make sure to follow the formatting instructions precisely.

# Input Description:
1. Current assigned subtask: The specific task you're currently working on, provided by the system.
2. Observation of current step: A description of the current state of the web page, represented as an accessibility tree (AXTree).
3. History of interactions: The previous actions you have taken and brief explanations for each.
4. Page Interaction Log: A summary of the interactions that have occurred on the page, including the controls that have been interacted with.
5. Action space: The list of possible actions you can take. The chosen action must conform to this predefined action space and be executable by the program. There are 12 different types of actions available:

---
noop(wait_ms: float = 1000)
- Description: Do nothing, and optionally wait for the given time (in milliseconds).
- Examples:
  - `noop()`
  - `noop(500)`

scroll(delta_x: float, delta_y: float)
- Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
- Examples:
  - `scroll(0, 200)`

fill(bid: str, value: str)
- Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for `<input>`, `<textarea>` and `[contenteditable]` elements.
- Examples:
  - `fill('237', 'example value')`

select_option(bid: str, options: str | list[str])
- Description: Select one or multiple options in a `<select>` element. You can specify option value or label to select. Multiple options can be selected.
- Examples:
  - `select_option('a48', 'blue')`
  - `select_option('c48', ['red', 'green', 'blue'])`

click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
- Description: Click an element.
- Examples:
  - `click('a51')`
  - `click('b22', button='right')`
  - `click('48', button='middle', modifiers=['Shift'])`

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
- Description: Double-click an element.
- Examples:
  - `dblclick('12')`
  - `dblclick('ca42', button='right')`
  - `dblclick('178', button='middle', modifiers=['Shift'])`

hover(bid: str)
- Description: Hover over an element.
- Examples:
  - `hover('b8')`

press(bid: str, key_comb: str)
- Description: Focus the matching element and press a combination of keys. Accepts logical key names emitted in the `keyboardEvent.key` property of keyboard events.
- Examples:
  - `press('88', 'Backspace')`
  - `press('a26', 'ControlOrMeta+a')`
  - `press('a61', 'Meta+Shift+t')`

focus(bid: str)
- Description: Focus the matching element.
- Examples:
  - `focus('b455')`

clear(bid: str)
- Description: Clear the input field.
- Examples:
  - `clear('996')`

drag_and_drop(from_bid: str, to_bid: str)
- Description: Perform a drag & drop operation.
- Examples:
  - `drag_and_drop('56', '498')`

upload_file(bid: str, file: str | list[str])
- Description: Click an element and upload one or multiple files.
- Examples:
  - `upload_file('572', 'my_receipt.pdf')`
  - `upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])`
  
---

# Steps

1. Analyze the Input: Review the chat messages, the current assigned subtask, the AXTree, and the history of interactions to understand the context and determine what needs to be done next.
2. Reasoning: Think step-by-step about the effects of your previous actions, the current state of the page, and what action is needed to progress towards the goal. Your reasoning should align the task intent with the chosen action and explain why you are selecting a specific control or element.
3. Determine the Next Action: Based on your reasoning, choose the most appropriate next action from the action space. Ensure the action is correctly formatted and includes all necessary parameters.

# Output Format

Your response should include only the following two sections, formatted exactly as specified:
1. <reflection>...</reflection>
   - Purpose: Summarize the effects of past actions and their impact on your decision-making.
   - Format: `<reflection>Your reflection here.</reflection>`
   - Content: Highlight key outcomes from previous actions, lessons learned, and how they guide the next step. Avoid action redundancy.
   
2. <think>...</think>
   - Purpose: Explain your reasoning for the next action.
   - Format: `<think>Your reasoning here.</think>`
   - Content: Describe, step-by-step, how you arrived at the action. Detail why the chosen action is appropriate based on the current state and your reflection. State what you expect to achieve. Ensure it aligns with the <reflection> section.

2. <action>...</action>
   - Purpose: Specify the next action to take.
   - Format: `<action>Your chosen action here.</action>`
   - Content: Use exact syntax and parameters from the action space to define your action. Ensure it aligns with the <think> section.

Important Notes:

- Reasoning Before Action: Give out the 'reflection' section firstly, then always include the 'think' section before the 'action' section. Do not start with the conclusion or action.
- Clarity and Specificity: Be precise and detailed in your reasoning. Explicitly describe the alignment between the task intent and the chosen action, especially why you choose the specific control.
- Formatting: Do not include any additional text outside of the specified sections. Do not add any code blocks, markdown formatting, or extraneous information.
- Output Only: Only output the '<reflection>', '<think>', '<action>' sections as specified. Do not include any headers, footers, or additional commentary.

# Example

Input:

1. Current assigned subtask: Log in the user account on OpenStreetMap.
2. Observation of current step:

```AXTree
   RootWebArea 'OpenStreetMap', focused
   - [49] banner ''
     - [50] heading 'OpenStreetMap logo OpenStreetMap'
       - [51] link 'OpenStreetMap logo OpenStreetMap'
         - [52] image 'OpenStreetMap logo'
     - [54] navigation ''
       - [104] link 'Edit'
       - [111] link 'History'
       - [112] link 'Export'
     - [113] navigation ''
       - [114] list ''
         - [115] listitem ''
           - [116] link 'GPS Traces'
         - [117] listitem ''
           - [118] link 'User Diaries'
         - [119] listitem ''
           - [120] link 'Communities'
         - [121] listitem ''
           - [122] link 'Copyright'
         - [123] listitem ''
           - [124] link 'Help'
         - [125] listitem ''
           - [126] link 'About'
       - [143] link 'Log In'
       - [144] link 'Sign Up'
   - [148] Section ''
     - [152] textbox 'Search', focused
     - [154] button 'Where is this?'
     - [155] button 'Go'
     - [158] link 'Find directions between two points'
   - [202] heading 'Welcome to OpenStreetMap!'
   - [204] button 'Close'
```

4. History of interactions:

   ## Step 0
   ### Action:
   scroll(0, 200)
   ### Explanation:
   The assistant need to scroll down the page to find the login form, in order to log in the user account.

   ## Step 1
   ### Action:
   scroll(0, 200)
   ### Explanation:
   The assistant need to scroll down further the page to find the login form, in order to log in the user account.
   
5. Current Page Interaction Log:
- Scrolled: scroll(0, 200), scroll(0, 200)

Output:

<reflection>
The page has been scrolled twice, and the 'Log In' link (bid '143') is now visible. No further scrolling is needed as the desired element is accessible.
</reflection>
<think>
To proceed with the login process, I will click the 'Log In' link (bid '143'). This will navigate to the login form and advance the task.
</think>
<action>
click('143')
</action>

# Notes

- Remember to align your reasoning with the task intent and explicitly state why you are choosing a particular action.
- Ensure that your action is selected from the action space and is formatted correctly.
- Do not include any other text or formatting outside of the '<reflection>', '<think>' and '<action>' sections.
- Please consider interaction history to avoid repetitive and ineffective actions 


Given the following input, please generate think and action accordingly.


Input:

1. Current assigned subtask: Navigate to "Best-sellers" Section

2. Observation of current step:
```AXTree
RootWebArea 'Dashboard / Magento Admin', focused
	[130] link 'Magento Admin Panel'
		[131] image 'Magento Admin Panel'
	[132] navigation ''
		[133] menubar '', orientation='horizontal'
			[135] link '\ue604 DASHBOARD'
			[138] link '\ue60b SALES'
			[171] link '\ue608 CATALOG'
			[189] link '\ue603 CUSTOMERS'
			[208] link '\ue609 MARKETING'
			[275] link '\ue602 CONTENT'
			[321] link '\ue60a REPORTS'
			[446] link '\ue60d STORES'
			[522] link '\ue610 SYSTEM'
			[605] link '\ue612 FIND PARTNERS & EXTENSIONS'
	[624] banner ''
		[627] heading 'Dashboard'
		[630] link '\ue600 admin'
		[642] link '\ue607'
		[644] Section ''
			[646] LabelText ''
				StaticText '\ue60c'
			[650] textbox '\ue60c'
	[656] main ''
		StaticText 'Scope:'
		[667] button 'All Store Views', focused, hasPopup='menu'
		[668] list ''
			[669] listitem ''
				StaticText 'All Store Views'
			[671] listitem ''
				StaticText 'Main Website'
			[673] listitem ''
				StaticText 'Main Website Store'
			[675] listitem ''
				[676] link 'Default Store View'
		[678] link '\ue633 What is this?'
		[683] Section ''
			[685] button 'Reload Data'
		[689] Section ''
			[691] HeaderAsNonLandmark ''
				StaticText 'Advanced Reporting'
			StaticText "Gain new insights and take command of your business' performance, using our dynamic product, order, and customer reports tailored to your customer data."
			[694] link 'Go to Advanced Reporting \ue644'
		StaticText 'Chart is disabled. To enable the chart, click'
		[701] link 'here'
		StaticText '.'
		[704] list ''
			[705] listitem ''
				StaticText 'Revenue'
				[707] strong ''
					StaticText '$0.00'
			[710] listitem ''
				StaticText 'Tax'
				[712] strong ''
					StaticText '$0.00'
			[715] listitem ''
				StaticText 'Shipping'
				[717] strong ''
					StaticText '$0.00'
			[720] listitem ''
				StaticText 'Quantity'
				[722] strong ''
					StaticText '0'
		[726] tablist '', multiselectable=False, orientation='horizontal'
			[727] tab 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers', expanded=True, selected=True, controls='grid_tab_ordered_products_content'
				[728] link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
			[734] tab 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products', expanded=False, selected=False, controls='ui-id-1'
				[735] link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products'
			[742] tab 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers', expanded=False, selected=False, controls='ui-id-2'
				[743] link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers'
			[750] tab 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers', expanded=False, selected=False, controls='ui-id-3'
				[751] link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers'
		[760] tabpanel 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
			[762] table ''
				[763] rowgroup ''
					[764] row ''
						[765] columnheader 'Product'
						[767] columnheader 'Price'
						[769] columnheader 'Quantity'
				[771] rowgroup ''
					[772] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/20/'
						[773] gridcell 'Quest Lumaflex™ Band'
						[774] gridcell '$19.00'
						[775] gridcell '6'
					[776] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/29/'
						[777] gridcell 'Sprite Stasis Ball 65 cm'
						[778] gridcell '$27.00'
						[779] gridcell '6'
					[780] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/25/'
						[781] gridcell 'Sprite Stasis Ball 55 cm'
						[782] gridcell '$23.00'
						[783] gridcell '5'
					[784] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/13/'
						[785] gridcell 'Overnight Duffle'
						[786] gridcell '$45.00'
						[787] gridcell '5'
					[788] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/33/'
						[789] gridcell 'Sprite Yoga Strap 6 foot'
						[790] gridcell '$14.00'
						[791] gridcell '5'
		StaticText 'Lifetime Sales'
		[799] strong ''
			StaticText '$0.00'
		StaticText 'Average Order'
		[805] strong ''
			StaticText '$0.00'
		StaticText 'Last Orders'
		[811] table ''
			[812] rowgroup ''
				[813] row ''
					[814] columnheader 'Customer'
					[816] columnheader 'Items'
					[818] columnheader 'Total'
			[820] rowgroup ''
				[821] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/299/'
					[822] gridcell 'Sarah Miller'
					[823] gridcell '5'
					[824] gridcell '$194.40'
				[825] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/65/'
					[826] gridcell 'Grace Nguyen'
					[827] gridcell '4'
					[828] gridcell '$190.00'
				[829] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/125/'
					[830] gridcell 'Matt Baker'
					[831] gridcell '3'
					[832] gridcell '$151.40'
				[833] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/136/'
					[834] gridcell 'Lily Potter'
					[835] gridcell '4'
					[836] gridcell '$188.20'
				[837] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/230/'
					[838] gridcell 'Ava Brown'
					[839] gridcell '2'
					[840] gridcell '$83.40'
		StaticText 'Last Search Terms'
		[845] table ''
			[846] rowgroup ''
				[847] row ''
					[848] columnheader 'Search Term'
					[850] columnheader 'Results'
					[852] columnheader 'Uses'
			[854] rowgroup ''
				[855] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/13/'
					[856] gridcell '100'
					[857] gridcell '23'
					[858] gridcell '2'
				[859] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/25/'
					[860] gridcell 'tanks'
					[861] gridcell '23'
					[862] gridcell '1'
				[863] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/19/'
					[864] gridcell 'nike'
					[865] gridcell '0'
					[866] gridcell '3'
				[867] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/1/'
					[868] gridcell 'Joust Bag'
					[869] gridcell '10'
					[870] gridcell '4'
				[871] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/11/'
					[872] gridcell 'hollister'
					[873] gridcell '1'
					[874] gridcell '19'
		StaticText 'Top Search Terms'
		[879] table ''
			[880] rowgroup ''
				[881] row ''
					[882] columnheader 'Search Term'
					[884] columnheader 'Results'
					[886] columnheader 'Uses'
			[888] rowgroup ''
				[889] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/11/'
					[890] gridcell 'hollister'
					[891] gridcell '1'
					[892] gridcell '19'
				[893] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/1/'
					[894] gridcell 'Joust Bag'
					[895] gridcell '10'
					[896] gridcell '4'
				[897] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/13/'
					[898] gridcell '100'
					[899] gridcell '23'
					[900] gridcell '2'
				[901] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/25/'
					[902] gridcell 'tanks'
					[903] gridcell '23'
					[904] gridcell '1'
				[905] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/9/'
					[906] gridcell 'WP10'
					[907] gridcell '1'
					[908] gridcell '1'
	[910] contentinfo ''
		[913] paragraph ''
			[914] link '\ue606'
			StaticText 'Copyright © 2024 Magento Commerce Inc. All rights reserved.'
		[916] paragraph ''
			[917] strong ''
				StaticText 'Magento'
			StaticText 'ver. 2.4.6'
		[918] link 'Privacy Policy'
		StaticText '|'
		[919] link 'Account Activity'
		StaticText '|'
		[920] link 'Report an Issue'
```

3. History of interactions:

## step 0

### Action:
click('728')

### Explanation:
The assistant checked the content under the 'Reports' section to know what he can do. '728' is target control id.


## step 1

### Action:
click('727')

### Explanation:
The assistant clicked the content under the 'best-selling' section. '727' is target control id. However, the assistant can't find the valid time range in January 2023. I may need to navigate to another page to find the information.

## step 2

### Action:
scroll(0, 200)

### Explanation:
The assistant scrolled the page, but didn't find anything relevant to time range in January 2023.

## step 3

### Action:
scroll(0, 200)

### Explanation:
The assistant scrolled further the page, but didn't find anything relevant to time range in January 2023.

4. Page Interaction Log:

- Clicked: click('728'), click('727')
- Scrolled: scroll(0, 200), scroll(0, 200)

"""
        }
    ]
    
    start = time()
    
    # see: original type 1
    r = client.chat.completions.create(
        model=model,
        messages=payload,
        temperature=1.0,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        n=20,
        logprobs=True,
        top_logprobs=5,
    )
    
    print(f"\nTime taken: {time() - start:.2f}s")
    
    def confidence_score(c):
        """return confidence score following macro-r1 metric.

        Args:
            c (List[ChatCompletionTokenLogprob]): list of probabilities for each token in the completion.
        """
        results = [math.exp(e.logprob) / sum([math.exp(v.logprob) for v in e.top_logprobs]) for e in c.logprobs.content]
        return sum(results) / len(results)
    
    previous_actions = ["click('728')", "click('727')", "scroll(0, 200)"]
    count = 0
    total = len(r.choices)
    
    with open('out.txt', 'w', encoding='utf-8') as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(f):
            for c in r.choices:
                cs = confidence_score(c)
                print(f"\nconfidence_score: {cs:.2f},response:\n {c.message.content}")
                if any([a in c.message.content for a in previous_actions]):
                    count += 1
            print(f"\nRepeated action number: {count}, total: {total}")
