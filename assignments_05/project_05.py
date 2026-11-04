from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

# Task 1: Setup and System Prompt
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """You are an expert Job Application Coach. Your specific purpose is to assist professionals 
transitioning into data engineeringrole —by helping them translate their past experience and 
recent coursework into compelling resumes, cover letters, and portfolio descriptions.

Follow these strict constraints:
1. Stay rigidly focused on job application materials.
    1.1 Help create and improve resumes, cover letters, professional summaries, LinkedIn profiles, and application responses.
    1.2 Help tailor application materials to specific job descriptions.
    1.3 Identify relevant skills, experience, and transferable qualifications.
    1.4 Help translate technical experience into clear, measurable, and results-oriented accomplishments.
    1.5 Suggest relevant keywords for applicant tracking systems (ATS).
    1.6 Help prepare for job interviews by developing answers to behavioral and technical questions.
2. Acknowledge that you may not know the highly specific, rapidly changing industry norms for modern data stacks or tech hiring. 
Always advise the user to apply their own professional judgment to your suggestions.
3. Adapt to Individual Job Opportunities
    3.1 Analyze job descriptions to identify required skills, preferred qualifications, responsibilities, and potential gaps.
    3.2 Recommend doable portfolio projects for the job
    3.3 Help the user position their QA and software engineering background as an advantage when transitioning into data and AI-related careers.
4. Always conclude your advice by reminding the user to carefully review, edit, and fact-check the generated text 
to ensure it reflects their authentic experience and voice before submitting it anywhere.
"""
# I several deliberate choices when writing this system prompt on of them is 
# specified the career transition. Rather than creating a generic job application assistant, 
# I defined the user as someone transitioning from QA and software engineering into data engineering role.
# This helps the model recognize transferable skills and recommend relevant portfolio projects.

# Task 2: Bullet Point Rewriter
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Respond with a single JSON object. For example:
    "original": "worked on customer cases"
    "improved": "you answer here that contains improved version of the original text"
    Do that for every bullet and return clean JSON object that contains  key value pairs
    wrapped in square brackets.
    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    raw_response = get_completion(messages)
    clean_json = raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        rewritten_bullets = json.loads(clean_json)
        
        # Print both versions side by side
        print("--- Resume Bullet Point Revisions ---")
        for item in rewritten_bullets:
            print(f"Original : {item['original']}")
            print(f"Improved : {item['improved']}")
            
        return rewritten_bullets
        
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        print("Raw output:", raw_response)
        return []


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)
# These bullets are weak because they are overly generic, lack measurable metrics, 
# and fail to highlight the specific skills used. The model improved them by introducing 
# strong action verbs, specifying the exact nature of the work, and emphasizing the concrete business impact or end result.

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    print(response)
    return response


job_title = "Junior Data Engineer"
# background = "Five years of experience as a middle school math teacher; recently completed \
# a Python course and built data pipelines using Pandas."
background  = "8 years of experience as Quality assurance engineer and 5 years of Experience as " \
"Software engineer in test. Area of my expertise the availability of the networking systems." \
"Recently completed a Python course and build data pipeline using Pandas."
generate_cover_letter(job_title, background)
# Selected examples illustrate how to effectively pitch transferable skills during a career change. 
# Specifically, they demonstrate how to frame past non-technical domain experience 
# (like nursing or retail banking) as a unique, high-value asset for a new technical role, rather than a gap in experience.
# The few-shot pattern controls the structure, tone, and logical framework of the output.

# Task 4: Moderation Check
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    
    if flagged:
        print("Warning: The provided text was flagged for safety policy violations.")
        return False
        
    return True

safe_text = "I love taking my dog for a walk in the park on a sunny day."
print(f"Input: '{safe_text}'")
result_safe = is_safe(safe_text)
print(f"Final Result: {result_safe}\n")

flagged_text = "I want to hurt myself." 
print(f"Input: '{flagged_text}'")
result_flagged = is_safe(flagged_text)
print(f"Final Result: {result_flagged}")

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            rewritten_bullets = rewrite_bullets(raw_bullets)
            print(f"Bullets are: {rewritten_bullets}")

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            cover_letter = generate_cover_letter(job_title, background)
            print(f"Cover letter is: {cover_letter}")

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"Job Application Helper: {reply}")
            messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    run_chatbot()

# Task 6: Ethics Reflection
# Question: Your bot was trained on text written by and about certain kinds of people. 
# How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
# if a model is trained mostly on text written by and about certain kinds of people, it learns their patterns as the default. 
# Groups that are overrepresented in the data shape what the model treats as normal, professional, or typical, 
# while underrepresented groups are modelled less accurately or more stereotypically.
# That can show up in advice in a few ways: communication styles: coltural backgrounds.

# What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
# AI-generated content should always be reviewed before submission. A model produces text based on probability — 
# the most plausible continuation — not on verified facts about the applicant, so it can confidently invent achievements, 
# metrics, dates, or skills. Submitting that unchecked risks sending false claims to an employer, 
# eing unable to defend them in an interview, and, if they are later contradicted, losing credibility or the offer itself. 

# What is one guardrail you would add if you were deploying this tool professionally? 
# (A guardrail is any design choice that reduces the chance of harm — a UI warning, 
# a moderation filter, a usage policy, a disclaimer, or something else entirely.)
