# AgenticAI_using_Gemma_For_VotersID_On_Local_Workstation



The objective of this project was to show to build an Agentic AI that uses a local voters database ( a CSV file ) and user can 
ask questions about the voters in the database.

To start with I created a voters database by going to the SIR Electoral rolls PDF file, and picked up a certain part number
and converted them to PNG files, and used Google Gemini API to parse the pngs to a CSV file.

The objective of this repo is not to show how to convert the PDF to PNG and later use OCR tool to create the CSV.

However, the objective is to show how we can use an external .csv file integrated to a PYTHON code that uses Gemma.


### Why Agentic AI 

```
Instead of writing and maintaining dozens of individual search scripts, your agent acts as a virtual data analyst sitting on top of your dataset—interpreting your intent, writing the code, running it locally, and giving you the exact rows you asked for.

How it works: It acts as an autonomous worker. It possesses reasoning loops (Plan $\rightarrow$ Execute $\rightarrow$ Observe $\rightarrow$ Repeat) and has access to tools (like a Python environment, web browser, or database).Analogy: It is like a digital intern or data analyst. You give it a goal (e.g., "Find all father-son voters in house #42"), and it figures out how to use tools (writing and running Pandas code on your CSV) to accomplish it.

```


Here you'll see the Python Code integrated with Gemma and uses or parses the .csv file and prompts the user for questions.

The questions that this Agentic AI will be useful to interpret the data in the voters db.

Some questions or prompts can be 

- Search for Duplicate Voters ID.

- Sort for any Duplicate Voters ID.

- Give me voters who are less than 18 years of age.

- Give me voters who are above 90 years of age.

A snap shot of what this Google's Gemma prompts is shown below. This is Gemma's possible prompts that user can give.

```
**Possible Tasks (and how I could approach them):**

1.  **Counting Unique Values:** I could count how many *unique* strings are in the list. This would give me a sense of the diversity of the identifiers.
2.  **Frequency Analysis:** I could count how many times each prefix (BCW, SOH, etc.) appears. This would reveal which category is most represented.
3.  **Filtering:** I could filter the list to show only strings that start with a specific prefix (e.g., all "BCW" identifiers).
4.  **Sorting:** I could sort the list alphabetically or numerically (if any of the strings contained numbers).
5.  **Identifying Potential Duplicates:**  I could check for repeated strings.  (Although, given the length of the list, this would be computationally intensive without further processing).

**To help me do something useful with this data, please tell me:**

```

My sample voters.csv  ( first line indicates the column names )


One can use any csv file to process data.

```
$ head voters.csv
source_page,part_number,serial_number,epic_number,name,relation_info,house_number,age,gender
page_003.png,1,1,SOH0467563,Name : Musthapha,Fathers Name: Hanif.M.K,01,61,Male
page_003.png,1,4,SOH5819115,Name : Nicholas Thekkumpuram Thomas,Others: Leena Thomas Thomas,1/B,53,Male
page_003.png,1,7,YTQ6290340,Name : Mahadevamma,Husbands Name: Mahalingaiah,2/1 Prashanth House,72,Female
page_003.png,1,10,SOH6171235,Name : SUCHITHRA C S,Fathers Name: SHIVANNA,3,21,Female
```

The voters.csv was from the SIR Electoral rolls 
<br>

<img width="862" height="283" alt="image" src="https://github.com/user-attachments/assets/7f977e2b-973d-4367-80a4c10ce637822e" />

<br>

## My approach :

### First

Installed Gemma3 on my local workstation
```
$ ollama list
NAME                 ID              SIZE      MODIFIED      
my-twin:latest       55049ae5c803    45 MB     12 days ago      
gemma3:4b            a2af6cc3eb7f    3.3 GB    11 months ago    
all-minilm:latest    1b226e2802db    45 MB     11 months ago

```

### Next created the csv file. One can use a similar structured data to accomplish this.


### Pip packages installed 

```
$ cat voter_agent.py 
import sys
import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

```

### Platform : Mac OS.


### Next, Python code 

```
 cat voter_agent.py 
import sys
import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 1. Load voter CSV
csv_filename = "voters.csv"
try:
    df = pd.read_csv(csv_filename)
    print(f"Successfully loaded '{csv_filename}' with {len(df)} records.")
except Exception as e:
    print(f"Error loading '{csv_filename}': {e}")
    sys.exit(1)

# 2. Connect to local Ollama (gemma3:4b)
llm = Ollama(
    model="gemma3:4b",
    temperature=0,
    num_predict=512  # Keeps step outputs concise for faster execution
)

# 3. Generalized System Prompt
PREFIX_PROMPT = """
You are a versatile electoral data analyst working with a pandas DataFrame named `df`.

Columns in `df`:
- `source_page`: PDF page number
- `part_number`: Polling booth/part number
- `serial_number`: Voter serial number
- `epic_number`: Unique Voter ID card number
- `name`: Voter's full name
- `relation_info`: Relation details (e.g., 'Father Name: ...', 'Husband Name: ...')
- `house_number`: Household or address identifier
- `age`: Voter age
- `gender`: Gender ('M', 'F', 'Male', 'Female', etc.)

CRITICAL INSTRUCTIONS:
1. Write Python/Pandas code to solve the user's request on `df`.
2. Execute the code to obtain the result.
3. As soon as you get the result, output your final answer immediately. Do not loop or execute repeatedly.
"""

# 4. Initialize Pandas Agent with iteration controls to prevent stopping errors
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,                   # Prints generated Pandas code to terminal
    allow_dangerous_code=True,
    max_iterations=20,              # Gives local model enough reasoning steps
    early_stopping_method="generate"# Forces a final summary if step limit is reached
)

def main():
    print("\n" + "=" * 60)
    print(" Electoral Data Agentic AI (Ollama - Gemma 3:4b)")
    print(" Ask any question about voters.csv!")
    print(" Type 'exit' or 'quit' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            # Prompt user for question
            user_question = input("\nAsk a question about voters.csv > ").strip()

            # Check exit commands
            if user_question.lower() in ["exit", "quit", "q"]:
                print("\nExiting Voter Agent. Goodbye!")
                break

            # Skip empty inputs
            if not user_question:
                continue

            # Run agent query
            print("\nProcessing request...\n")
            query = f"{PREFIX_PROMPT}\nUser Question: {user_question}"
            
            response = agent.invoke(query)
            
            # Extract final output text from response dictionary
            output_text = response.get("output", response) if isinstance(response, dict) else response

            print("\n--- Answer ---")
            print(output_text)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred while processing: {e}")

if __name__ == "__main__":
    main()

```


### Run - Some sample prompts

````

Ask a question about voters.csv > give me details of voters above 90 years

Processing request...

--- Answer ---
| source_page | part_number | serial_number | epic_number | name                | relation_info           | house_number | age | gender |
|---|---|---|---|---|---|---|---|---|
| page_011.png | 1 | 264 | 105 | SOH0467563 | Musthapha | Fathers Name: Hanif.M.K | 105 | 94 | Male |
| page_011.png | 1 | 267 | 113 | SOH5819115 | Nicholas Thekkumpuram Thomas | Others: Leena Thomas Thomas | 113 | 91 | Female |
------------------------------------------------------------


```

```

Ask a question about voters.csv > Give me the oldest Voter in this database ?

> Finished chain.

--- Answer ---
The oldest voter in this database is G.V. Anjanappa, who is 94 years old.
```
