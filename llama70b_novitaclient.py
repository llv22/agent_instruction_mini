from openai import OpenAI
import os
import math
from dotenv import load_dotenv

load_dotenv()

def confidence_score(c):
	"""return confidence score following macro-r1 metric.

	Args:
		c (List[ChatCompletionTokenLogprob]): list of probabilities for each token in the completion.
	"""
	results = [math.exp(e.logprob) / sum([math.exp(v.logprob) for v in e.top_logprobs]) for e in c.logprobs.content]
	return sum(results) / len(results)


if __name__ == "__main__":
    client = OpenAI(
        base_url="https://api.novita.ai/v3/openai",
        # Get the Novita AI API Key by referring to: https://novita.ai/docs/get-started/quickstart.html#_2-manage-api-key.
        api_key=os.environ["NOVITA_TOKEN"],
    )

    model = "meta-llama/llama-3.3-70b-instruct"
    stream = False  # or True
    max_tokens = 512

    chat_completion_res = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": """You are a UI Assistant. Your goal is to help the user perform tasks using a web browser. You communicate with the user via chat, where the user provides instructions and you can send messages back. You have access to a web browser that both you and the user can see, and only you can interact with via specific commands. Review the user's instructions, the current state of the web page, and the history of interactions to determine the best next action to accomplish your current subtask. Your response will be interpreted and executed by a program; make sure to follow the formatting instructions precisely.

# Input Description:
1. Current subtask: The specific task you're currently working on, provided by the system.
2. Observation of current step: A description of the current state of the web page, represented as an accessibility tree (AXTree).
3. History of interactions: The previous actions you have taken and brief explanations for each.
4. Page Interaction Log: A summary of the interactions that have occurred on the page and a short reflection based on action result.
5. Action space: The list of possible actions you can take. The chosen action must conform to this predefined action space and be executable by the program. There are 6 different types of actions available:

---
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

clear(bid: str)
- Description: Clear the input field.
- Examples:
  - `clear('996')`
  
complete(msg: str)
- Description: Specify the completion of current subtask and optionally send message if this task requires the result feedback, like extracting target information from the system.
- Examples:
    - `complete()`
    - `complete('the target products are: toll, monkey and cook')`
---

Only a single action can be provided at once. Example:
fill('a12', 'example with "quotes"')

# Steps

1. Analyze the Input: Review current assigned subtask, the AXTree, and the history of interactions to understand the context and determine what needs to be done next.
2. Reasoning: Think step-by-step about the screen regions available, the effects of your previous actions. Explore related regions and your thinking about how to move forward.
3. Determine the Next Action: Based on your reasoning, choose next action from the action space. Ensure the action is correctly formatted and includes all necessary parameters.

# Output Format

Your response should only output the '<screen_region>', '<reflection>', '<think>', '<action>' with the content specified as follow:
1. screen_region
   - Purpose: Briefly summarize key regions of the screen based on the current observation.
   - Format: `<screen_region>Your region analysis here.</screen_region>`
   - Content: Briefly categorize the screen region into different parts and summarize their functionalities.
   
2. reflection
   - Purpose: Briefly summarize the effects of past actions and their impact on your decision-making.
   - Format: `<reflection>Your reflection here.</reflection>`
   - Content: Check the section 'History of interactions' and think about why previous actions are effective or not, and avoid to repeated ineffective actions.
   
3. think
   - Purpose: Justify your selection of a screen region and the next action based on analysis.
   - Format: `<think>Your reasoning here.</think>`
   - Content: Describe, step-by-step, how you arrived at the action. First explore possible regions, then choose your action based on the current state and previous reflection. State what you expect to achieve. Ensure it aligns with the <reflection> section.

4. action
   - Purpose: Specify the next action to take.
   - Format: `<action>Your chosen action here.</action>`
   - Content: Use exact syntax and parameters from the action space to define your action. Ensure it aligns with the <think> section.

Important Notes:

- Reasoning Order: Split the screen into different regions in the 'screen_region' section, then give out the 'reflection' section, then always include the 'think' section before the 'action' section.
- Clarity and Specificity: Be precise and detailed in your reasoning. Explicitly describe the alignment between the task intent and the chosen action, especially why you choose the specific control.
- Formatting: Do not include any additional text outside of the specified sections. Do not add any code blocks, markdown formatting, or extraneous information.
- AXTree format: It describes the hierarchical structure of current web page. Each line has the structure `[bid] control_type 'control_value', extra_properties`. For non-interactive control, [bid] will be emitted.
- Action parameter: For the chose action that requires bid as parameter, ensure the bid is valid in AXTree.

# Your Task

Given the following input, Could you please generate the next action to take?

Input:

1. Current subtask: Navigate to the 'Best Seller' Section..
2. Observation of current step:

```AXTree
RootWebArea 'Dashboard / Magento Admin', focused
	[134] link+image 'Magento Admin Panel'
	[136] navigation ''
		[137] menubar '', orientation='horizontal'
			[139] link '\ue604 DASHBOARD'
			[142] link '\ue60b SALES'
			[175] link '\ue608 CATALOG'
			[193] link '\ue603 CUSTOMERS'
			[212] link '\ue609 MARKETING'
			[279] link '\ue602 CONTENT'
			[325] link '\ue60a REPORTS'
			[450] link '\ue60d STORES'
			[526] link '\ue610 SYSTEM'
			[609] link '\ue612 FIND PARTNERS & EXTENSIONS'
	banner ''
		heading 'Dashboard'
		[634] link '\ue600 admin'
		[646] link '\ue607'
		Section ''
			LabelText  '\ue60c'
			[654] textbox '\ue60c'
	main ''
		StaticText 'Scope:'
		[671] button 'All Store Views', hasPopup='menu'
		[682] link '\ue633 What is this?'
		Section ''
			[689] button 'Reload Data'
		Section ''
			HeaderAsNonLandmark  'Advanced Reporting'
			StaticText "Gain new insights and take command of your business' performance, using our dynamic product, order, and customer reports tailored to your customer data."
			[698] link 'Go to Advanced Reporting \ue644'
		StaticText 'Chart is disabled. To enable the chart, click'
		[705] link 'here'
		StaticText '.'
		list ''
			listitem ''
				StaticText 'Revenue'
				strong  '$0.00'
			listitem ''
				StaticText 'Tax'
				strong  '$0.00'
			listitem ''
				StaticText 'Shipping'
				strong  '$0.00'
			listitem ''
				StaticText 'Quantity'
				strong  '0'
		tablist '', multiselectable=False, orientation='horizontal'
			[731] tab+link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers', expanded=True, selected=True, controls='grid_tab_ordered_products_content'
			[738] tab+link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products', expanded=False, selected=False, controls='ui-id-1'
			[746] tab+link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers', expanded=False, selected=False, controls='ui-id-2'
			[754] tab+link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers', expanded=False, selected=False, controls='ui-id-3'
		tabpanel 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
			table ''
				rowgroup ''
					row ''
						columnheader 'Product'
						columnheader 'Price'
						columnheader 'Quantity'
				rowgroup ''
					[776] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/20/'
						gridcell 'Quest Lumaflex™ Band'
						gridcell '$19.00'
						gridcell '6'
					[780] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/29/'
						gridcell 'Sprite Stasis Ball 65 cm'
						gridcell '$27.00'
						gridcell '6'
					[784] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/33/'
						gridcell 'Sprite Yoga Strap 6 foot'
						gridcell '$14.00'
						gridcell '5'
					[788] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/25/'
						gridcell 'Sprite Stasis Ball 55 cm'
						gridcell '$23.00'
						gridcell '5'
					[792] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/catalog/product/edit/id/13/'
						gridcell 'Overnight Duffle'
						gridcell '$45.00'
						gridcell '5'
		StaticText 'Lifetime Sales'
		strong  '$0.00'
		StaticText 'Average Order'
		strong  '$0.00'
		StaticText 'Last Orders'
		table ''
			rowgroup ''
				row ''
					columnheader 'Customer'
					columnheader 'Items'
					columnheader 'Total'
			rowgroup ''
				[825] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/299/'
					gridcell 'Sarah Miller'
					gridcell '5'
					gridcell '$194.40'
				[829] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/65/'
					gridcell 'Grace Nguyen'
					gridcell '4'
					gridcell '$190.00'
				[833] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/125/'
					gridcell 'Matt Baker'
					gridcell '3'
					gridcell '$151.40'
				[837] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/136/'
					gridcell 'Lily Potter'
					gridcell '4'
					gridcell '$188.20'
				[841] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/sales/order/view/order_id/230/'
					gridcell 'Ava Brown'
					gridcell '2'
					gridcell '$83.40'
		StaticText 'Last Search Terms'
		table ''
			rowgroup ''
				row ''
					columnheader 'Search Term'
					columnheader 'Results'
					columnheader 'Uses'
			rowgroup ''
				[859] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/13/'
					gridcell '100'
					gridcell '23'
					gridcell '2'
				[863] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/25/'
					gridcell 'tanks'
					gridcell '23'
					gridcell '1'
				[867] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/19/'
					gridcell 'nike'
					gridcell '0'
					gridcell '3'
				[871] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/1/'
					gridcell 'Joust Bag'
					gridcell '10'
					gridcell '4'
				[875] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/11/'
					gridcell 'hollister'
					gridcell '1'
					gridcell '19'
		StaticText 'Top Search Terms'
		table ''
			rowgroup ''
				row ''
					columnheader 'Search Term'
					columnheader 'Results'
					columnheader 'Uses'
			rowgroup ''
				[893] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/11/'
					gridcell 'hollister'
					gridcell '1'
					gridcell '19'
				[897] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/1/'
					gridcell 'Joust Bag'
					gridcell '10'
					gridcell '4'
				[901] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/13/'
					gridcell '100'
					gridcell '23'
					gridcell '2'
				[905] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/25/'
					gridcell 'tanks'
					gridcell '23'
					gridcell '1'
				[909] row 'http://nlp-in-477-l.soe.ucsc.edu:7780/admin/search/term/edit/id/9/'
					gridcell 'WP10'
					gridcell '1'
					gridcell '1'
	contentinfo ''
		paragraph ''
			[918] link '\ue606'
			StaticText 'Copyright © 2024 Magento Commerce Inc. All rights reserved.'
		paragraph ''
			strong  'Magento'
			StaticText 'ver. 2.4.6'
		[922] link 'Privacy Policy'
		StaticText '|'
		[923] link 'Account Activity'
		StaticText '|'
		[924] link 'Report an Issue'
```

3. History of interactions for current subtask:

## step 0

### Action:
click('731')

### Explanation:
The assistant clicked the content under the 'most viewed product' section. '732' is target control id. However, only the information "We couldn't find any records." is displayed.
## step 1

### Action:
scroll(0, 200)

### Explanation:
The assistant scrolled the page, but didn't find anything relevant to time range in January 2023.
## step 2

### Action:
scroll(0, 200)

### Explanation:
The assistant scrolled further the page, but didn't find anything relevant to time range in January 2023.
## step 3

### Action:
click('776')

### Explanation:
The assistant click the item, but didn't find anything relevant to time range in January 2023.
## step 4

### Action:
click('731')

### Explanation:
The assistant checked the content under the 'Reports' section to know what he can do. '725' is target control id. But I can't find the 'Best Seller' section with target date range in January 2023.

4. Actions occurred on this page:

- click('731')
- scroll(0, 200)
- scroll(0, 200)
- click('776')
- click('731')
"""
            }
        ],
        stream=stream,
        max_tokens=max_tokens,    
        temperature=1.0,
        top_p=0.95,
        n=20,
        logprobs=True,
        top_logprobs=5,
        presence_penalty=1.0, # still OK for 31.41s
    )

    if stream:
        for chunk in chat_completion_res:
            print(chunk.choices[0].delta.content or "")
    else:
        if len(chat_completion_res.choices) == 1:
            print(chat_completion_res.choices[0].message.content)
        elif len(chat_completion_res.choices) > 1:
            for c in chat_completion_res.choices:
                cs = confidence_score(c)
                print(f"\nconfidence_score: {cs:.2f},response:\n {c.message.content}")