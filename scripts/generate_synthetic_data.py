#!/usr/bin/env python3
"""
Generate a synthetic Q&A dataset simulating employee org knowledge responses.

Creates ~640 examples across 10 fictional orgs in a semiconductor/tech context.
Each employee answers 8 standard org knowledge questions.

Org sizes vary to simulate real-world imbalance:
  - Some orgs have 1-2 employees responding
  - Others have 15-20 employees responding

Output: data/synthetic/raw_responses.jsonl
"""

import argparse
import json
import os
import random

# ---------------------------------------------------------------------------
# The 8 standard org knowledge questions
# ---------------------------------------------------------------------------
QUESTIONS = [
    "What does your team/organization do?",
    "What tools and software do you use daily?",
    "What are the biggest roadblocks or pain points in your work?",
    "What operating procedures does your team follow?",
    "Who do you collaborate with most, and for what?",
    "What knowledge is hardest to find or most often asked about?",
    "What would a new employee most need to know about your team?",
    "What processes have changed recently that people might not know about?",
]

# ---------------------------------------------------------------------------
# 10 fictional orgs in a semiconductor/tech company
# ---------------------------------------------------------------------------
ORGS = [
    {
        "name": "Silicon Design Engineering",
        "dept_code": "SDE",
        "focus": "RTL design and verification of next-gen SoC chips",
        "num_employees": 15,
        "tools": ["Cadence Virtuoso", "Synopsys VCS", "Git", "JIRA", "Confluence", "Slack"],
        "collaborators": ["Fabrication Ops", "Test Engineering", "Product Management"],
        "pain_points": ["long simulation runtimes", "outdated PDK libraries", "cross-site coordination"],
        "procedures": ["design reviews every 2 weeks", "tape-out checklist sign-off", "regression suite before merge"],
        "hard_knowledge": ["power grid constraints for new process nodes", "legacy IP block interfaces"],
        "new_employee_tips": ["get access to the internal wiki ASAP", "shadow a tape-out cycle before leading one"],
        "recent_changes": ["moved from SVN to Git last quarter", "new EDA license pool sharing policy"],
    },
    {
        "name": "Fabrication Operations",
        "dept_code": "FABOPS",
        "focus": "Managing wafer fabrication lines and yield optimization",
        "num_employees": 20,
        "tools": ["MES (Manufacturing Execution System)", "SPC tools", "Camstar", "SAP", "Python scripts"],
        "collaborators": ["Silicon Design Engineering", "Quality Assurance", "Equipment Maintenance"],
        "pain_points": ["unplanned equipment downtime", "yield excursions", "shift handoff communication gaps"],
        "procedures": ["daily yield standup at 7am", "lot disposition within 4 hours", "contamination protocol"],
        "hard_knowledge": ["recipe tuning parameters for new tools", "historical yield data by product"],
        "new_employee_tips": ["cleanroom certification takes 2 weeks", "always check the interlock status board"],
        "recent_changes": ["new automated defect classification system", "updated gowning procedure for Bay 3"],
    },
    {
        "name": "Test Engineering",
        "dept_code": "TE",
        "focus": "Post-silicon validation and production testing of chips",
        "num_employees": 10,
        "tools": ["Teradyne ATE", "Python", "LabVIEW", "Splunk", "TestStand", "Confluence"],
        "collaborators": ["Silicon Design Engineering", "Fabrication Operations", "Customer Engineering"],
        "pain_points": ["test program debug turnaround", "limited ATE time slots", "correlation between bench and ATE"],
        "procedures": ["test plan review before first silicon", "guardbanding sign-off", "weekly failure pareto review"],
        "hard_knowledge": ["test coverage gaps per product line", "ATE loadboard schematics"],
        "new_employee_tips": ["learn the test program framework first", "the ATE lab has its own badge access"],
        "recent_changes": ["migrated test data to Splunk dashboards", "new trim algorithm for analog parts"],
    },
    {
        "name": "Product Management",
        "dept_code": "PM",
        "focus": "Defining product roadmaps and customer requirements for chip products",
        "num_employees": 2,
        "tools": ["Aha!", "Salesforce", "Confluence", "Excel", "PowerPoint", "Slack"],
        "collaborators": ["Silicon Design Engineering", "Sales", "Customer Engineering", "Marketing"],
        "pain_points": ["competing customer priorities", "long development cycles vs. market timing", "requirements churn"],
        "procedures": ["quarterly roadmap review", "MRD sign-off before design kickoff", "win/loss analysis"],
        "hard_knowledge": ["competitive landscape details", "customer-specific NDA requirements"],
        "new_employee_tips": ["read the last 3 MRDs to understand product direction", "build relationships with design leads early"],
        "recent_changes": ["new product tiering strategy", "moved roadmap planning from Excel to Aha!"],
    },
    {
        "name": "DevOps & IT Infrastructure",
        "dept_code": "DEVOPS",
        "focus": "Internal tooling, CI/CD pipelines, EDA compute infrastructure",
        "num_employees": 8,
        "tools": ["Kubernetes", "Terraform", "Jenkins", "Ansible", "Grafana", "PagerDuty", "AWS", "Slurm"],
        "collaborators": ["Silicon Design Engineering", "Test Engineering", "Security", "all engineering teams"],
        "pain_points": ["EDA license bottlenecks", "legacy on-prem to cloud migration", "sprawling Jenkins pipelines"],
        "procedures": ["change management board approval for prod", "on-call rotation weekly", "post-incident review within 48h"],
        "hard_knowledge": ["network topology for EDA farms", "license server configuration"],
        "new_employee_tips": ["get added to PagerDuty rotations in week 2", "the internal wiki has runbooks for common incidents"],
        "recent_changes": ["migrating EDA workloads to hybrid cloud", "new SSO provider rollout"],
    },
    {
        "name": "Quality Assurance",
        "dept_code": "QA",
        "focus": "Reliability testing, failure analysis, and compliance for shipped products",
        "num_employees": 5,
        "tools": ["Minitab", "JMP", "SAP QM", "SEM/TEM imaging tools", "Excel", "SharePoint"],
        "collaborators": ["Fabrication Operations", "Test Engineering", "Customer Engineering", "Regulatory"],
        "pain_points": ["slow failure analysis turnaround", "tracking corrective actions across teams", "audit prep overhead"],
        "procedures": ["8D process for customer returns", "weekly reliability review board", "PPAP for new products"],
        "hard_knowledge": ["failure mode libraries by product family", "JEDEC qualification requirements"],
        "new_employee_tips": ["the 8D template is on SharePoint, not Confluence", "shadow an FA engineer your first week"],
        "recent_changes": ["new automotive qualification requirements (AEC-Q100 Rev H)", "moved CAPA tracking to SAP QM"],
    },
    {
        "name": "Customer Engineering",
        "dept_code": "CE",
        "focus": "Technical support and integration help for key customers using our chips",
        "num_employees": 6,
        "tools": ["Zendesk", "Salesforce", "Confluence", "Slack", "oscilloscopes", "logic analyzers"],
        "collaborators": ["Product Management", "Test Engineering", "Silicon Design Engineering", "Sales"],
        "pain_points": ["reproducing customer-reported issues in-house", "NDA barriers to sharing debug info", "travel scheduling"],
        "procedures": ["SLA: respond within 4 hours for P1", "escalation path to design team", "trip report within 3 days"],
        "hard_knowledge": ["customer board designs and constraints", "undocumented chip errata workarounds"],
        "new_employee_tips": ["learn the top 5 customer accounts first", "the errata list is more useful than the datasheet"],
        "recent_changes": ["new customer portal for self-service docs", "restructured P1/P2/P3 severity definitions"],
    },
    {
        "name": "HR & People Operations",
        "dept_code": "HR",
        "focus": "Hiring, onboarding, benefits, and employee experience for the org",
        "num_employees": 1,
        "tools": ["Workday", "Greenhouse", "Slack", "Google Workspace", "Culture Amp"],
        "collaborators": ["all departments", "Finance", "Legal", "hiring managers"],
        "pain_points": ["slow approvals for headcount", "inconsistent onboarding across sites", "benefits FAQ overload"],
        "procedures": ["new hire onboarding checklist", "quarterly engagement survey", "annual performance review cycle in Q4"],
        "hard_knowledge": ["visa sponsorship policies", "equity vesting schedule details"],
        "new_employee_tips": ["Workday is the single source of truth for HR", "the benefits guide is updated every January"],
        "recent_changes": ["switched to Greenhouse for recruiting", "new hybrid work policy effective Q1"],
    },
    {
        "name": "Embedded Software",
        "dept_code": "ESW",
        "focus": "Firmware and SDK development for our chip products",
        "num_employees": 12,
        "tools": ["VS Code", "GCC ARM", "JTAG debuggers", "Git", "JIRA", "Jenkins", "QEMU"],
        "collaborators": ["Silicon Design Engineering", "Customer Engineering", "Test Engineering"],
        "pain_points": ["hardware bring-up delays", "fragmented driver codebase", "documentation lag behind silicon changes"],
        "procedures": ["code review required before merge", "weekly firmware sync", "release branching for each chip revision"],
        "hard_knowledge": ["register maps for pre-release silicon", "boot ROM quirks per chip revision"],
        "new_employee_tips": ["start with the SDK getting-started guide", "the JTAG setup wiki page is outdated — ask Sam"],
        "recent_changes": ["adopted Zephyr RTOS for new products", "new CI pipeline runs hardware-in-the-loop tests"],
    },
    {
        "name": "Supply Chain & Procurement",
        "dept_code": "SCP",
        "focus": "Sourcing raw materials, managing foundry relationships, and logistics",
        "num_employees": 1,
        "tools": ["SAP MM", "Oracle", "Excel", "Power BI", "email (lots of it)"],
        "collaborators": ["Fabrication Operations", "Finance", "external foundry partners", "Quality Assurance"],
        "pain_points": ["lead time volatility for specialty chemicals", "single-source supplier risk", "demand forecast accuracy"],
        "procedures": ["dual-source policy for critical materials", "quarterly business review with foundries", "PO approval chain"],
        "hard_knowledge": ["supplier contract terms and pricing", "alternative material qualifications"],
        "new_employee_tips": ["the SAP material master data is a mess — always cross-check", "build relationships with foundry contacts early"],
        "recent_changes": ["new supplier diversity requirements", "moved demand planning to Power BI from Excel"],
    },
]

# ---------------------------------------------------------------------------
# Response templates — varied phrasing to feel natural
# ---------------------------------------------------------------------------

def _make_response_variants():
    """Return a dict mapping question index to a list of template functions.
    Each function takes (org, employee_name, employee_role) and returns a response string.
    """
    variants = {}

    # Q0: What does your team/organization do?
    variants[0] = [
        lambda o, n, r: f"I'm on the {o['name']} team. We focus on {o['focus']}. My role as {r} means I'm mostly involved in the day-to-day execution of that mission.",
        lambda o, n, r: f"Our team, {o['name']} ({o['dept_code']}), is responsible for {o['focus']}. It's a critical function because our output feeds directly into what {o['collaborators'][0]} and {o['collaborators'][1]} do.",
        lambda o, n, r: f"In short, {o['focus']}. The {o['name']} group is about {random.choice(['6', '8', '12', '15', '20', '25', '30'])} people and we sit under the engineering org.",
        lambda o, n, r: f"I work in {o['dept_code']}. We handle {o['focus']}. As a {r}, I spend most of my time on the technical side of that.",
    ]

    # Q1: What tools and software do you use daily?
    variants[1] = [
        lambda o, n, r: f"My daily toolkit: {', '.join(o['tools'][:4])}. I also use {o['tools'][-1]} pretty regularly but not every single day.",
        lambda o, n, r: f"Honestly, I live in {o['tools'][0]} and {o['tools'][1]}. We also rely on {', '.join(o['tools'][2:])} for communication and tracking.",
        lambda o, n, r: f"The main ones are {', '.join(o['tools'])}. If I had to pick the one I couldn't live without, it'd be {random.choice(o['tools'][:3])}.",
        lambda o, n, r: f"For my {r} work, I'm mainly in {o['tools'][0]}. The team as a whole uses {', '.join(o['tools'][1:])} too.",
    ]

    # Q2: What are the biggest roadblocks or pain points?
    variants[2] = [
        lambda o, n, r: f"The biggest pain point is definitely {o['pain_points'][0]}. We also struggle with {o['pain_points'][1]}, which slows us down more than people realize.",
        lambda o, n, r: f"Where do I start? {o['pain_points'][0].capitalize()} is a constant battle. {o['pain_points'][2].capitalize()} is also a recurring issue that we've been escalating.",
        lambda o, n, r: f"I'd say: 1) {o['pain_points'][0]}, 2) {o['pain_points'][1]}, and 3) {o['pain_points'][2]}. The first one is the most impactful on our delivery timelines.",
        lambda o, n, r: f"From my perspective as a {r}, {o['pain_points'][1]} is the thing that burns the most time. The team overall would probably also mention {o['pain_points'][0]}.",
    ]

    # Q3: What operating procedures does your team follow?
    variants[3] = [
        lambda o, n, r: f"Our main procedures are: {o['procedures'][0]}, {o['procedures'][1]}, and {o['procedures'][2]}. These are pretty strictly followed.",
        lambda o, n, r: f"We follow a few key processes. The most important is probably {o['procedures'][0]}. We also do {o['procedures'][1]} which keeps things from falling through the cracks.",
        lambda o, n, r: f"The non-negotiables are {o['procedures'][1]} and {o['procedures'][2]}. We also have {o['procedures'][0]} but that's more of a team norm than a hard rule.",
        lambda o, n, r: f"As a {r}, the procedure I interact with most is {o['procedures'][0]}. The team also maintains {o['procedures'][1]} and {o['procedures'][2]}.",
    ]

    # Q4: Who do you collaborate with most?
    variants[4] = [
        lambda o, n, r: f"We work most closely with {o['collaborators'][0]} — that's almost a daily interaction. {o['collaborators'][1]} is also a frequent collaborator, especially during crunch periods.",
        lambda o, n, r: f"My top collaborators are {o['collaborators'][0]} and {o['collaborators'][1]}. We also loop in {o['collaborators'][-1]} when needed but it's less frequent.",
        lambda o, n, r: f"{', '.join(o['collaborators'])}. The nature of {o['focus']} means we're always coordinating with other teams. {o['collaborators'][0]} is the most critical dependency.",
        lambda o, n, r: f"In my {r} role, I interface a lot with {o['collaborators'][0]}. The broader team also works with {o['collaborators'][1]} and {o['collaborators'][-1]}.",
    ]

    # Q5: What knowledge is hardest to find?
    variants[5] = [
        lambda o, n, r: f"The hardest thing to find info on is {o['hard_knowledge'][0]}. It's mostly tribal knowledge held by a few senior people. {o['hard_knowledge'][1]} is also poorly documented.",
        lambda o, n, r: f"Definitely {o['hard_knowledge'][0]}. People ask about it constantly and the docs are either outdated or scattered across Confluence, email threads, and people's heads.",
        lambda o, n, r: f"Two things: {o['hard_knowledge'][0]} and {o['hard_knowledge'][1]}. For the first one, you basically have to know who to ask. There's no single source of truth.",
        lambda o, n, r: f"As a relatively {random.choice(['new', 'recent'])} {r}, I found {o['hard_knowledge'][1]} the hardest to track down. Experienced folks know it intuitively but it's not written anywhere useful.",
    ]

    # Q6: What would a new employee need to know?
    variants[6] = [
        lambda o, n, r: f"First thing: {o['new_employee_tips'][0]}. Also, {o['new_employee_tips'][1]}. Beyond that, just ask questions — everyone on the team is approachable.",
        lambda o, n, r: f"I'd tell them: {o['new_employee_tips'][0]}. And honestly, {o['new_employee_tips'][1]}. The learning curve is steep but manageable if you're proactive.",
        lambda o, n, r: f"The most important thing is {o['new_employee_tips'][0]}. I wish someone had told me that on day one. {o['new_employee_tips'][1]} is also key.",
        lambda o, n, r: f"Two pieces of advice: {o['new_employee_tips'][0]}, and {o['new_employee_tips'][1]}. The {o['dept_code']} team is friendly — don't be afraid to bug people with questions.",
    ]

    # Q7: What processes have changed recently?
    variants[7] = [
        lambda o, n, r: f"The big one is: {o['recent_changes'][0]}. That's been a significant shift for us. Also, {o['recent_changes'][1]} — that caught some people off guard.",
        lambda o, n, r: f"Recently we {o['recent_changes'][0].lower() if not o['recent_changes'][0][0].isupper() else o['recent_changes'][0]}. It's still settling in. {o['recent_changes'][1]} is the other notable change.",
        lambda o, n, r: f"Two major changes: {o['recent_changes'][0]}, and {o['recent_changes'][1]}. The first one has been mostly positive. The second one is still being figured out.",
        lambda o, n, r: f"If you haven't heard: {o['recent_changes'][0]}. That affects how we do things daily. {o['recent_changes'][1]} is also new but less impactful for most people.",
    ]

    return variants


# Employee name pool
FIRST_NAMES = [
    "Alex", "Jordan", "Casey", "Morgan", "Taylor", "Riley", "Quinn", "Avery",
    "Sam", "Jamie", "Drew", "Blake", "Reese", "Skyler", "Dakota", "Cameron",
    "Finley", "Rowan", "Harper", "Emerson", "Parker", "Sage", "Hayden", "Ellis",
    "Charlie", "Kai", "Noel", "Remy", "Jules", "Arden", "Lane", "Shay",
    "Tatum", "Lennox", "Phoenix", "River", "Wren", "Oakley", "Marlowe", "Sutton",
]

ROLES_BY_ORG = {
    "SDE": ["RTL Design Engineer", "Verification Engineer", "Physical Design Engineer", "Design Lead", "Senior Verification Engineer"],
    "FABOPS": ["Process Engineer", "Yield Engineer", "Shift Supervisor", "Equipment Engineer", "Senior Process Engineer", "Fab Technician"],
    "TE": ["Test Engineer", "Test Program Developer", "ATE Technician", "Senior Test Engineer", "Test Lead"],
    "PM": ["Product Manager", "Senior Product Manager"],
    "DEVOPS": ["DevOps Engineer", "Site Reliability Engineer", "Systems Administrator", "Cloud Infrastructure Engineer", "DevOps Lead"],
    "QA": ["Reliability Engineer", "Failure Analysis Engineer", "Quality Engineer", "QA Lead"],
    "CE": ["Customer Applications Engineer", "Field Applications Engineer", "Senior Applications Engineer", "CE Lead"],
    "HR": ["HR Business Partner"],
    "ESW": ["Firmware Engineer", "SDK Developer", "Embedded Software Lead", "Senior Firmware Engineer", "BSP Engineer"],
    "SCP": ["Procurement Specialist"],
}


def generate_dataset(output_path: str, seed: int = 42):
    """Generate the full synthetic dataset."""
    random.seed(seed)
    variants = _make_response_variants()

    records = []
    used_names = set()

    for org in ORGS:
        dept = org["dept_code"]
        num_emp = org["num_employees"]
        available_roles = ROLES_BY_ORG[dept]

        # Pick unique employee names for this org
        available_names = [n for n in FIRST_NAMES if n not in used_names]
        if len(available_names) < num_emp:
            # Reset if we run out (unlikely with 40 names and ~80 total employees)
            available_names = list(FIRST_NAMES)
        emp_names = random.sample(available_names, min(num_emp, len(available_names)))
        used_names.update(emp_names)

        for emp_idx, emp_name in enumerate(emp_names):
            role = available_roles[emp_idx % len(available_roles)]

            for q_idx, question in enumerate(QUESTIONS):
                # Pick a random variant for this question
                template_fn = random.choice(variants[q_idx])
                response = template_fn(org, emp_name, role)

                record = {
                    "org": org["name"],
                    "dept_code": dept,
                    "employee_name": emp_name,
                    "employee_role": role,
                    "question": question,
                    "response": response,
                }
                records.append(record)

    # Shuffle to mix orgs together
    random.shuffle(records)

    # Write to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Print summary
    print(f"Generated {len(records)} examples")
    print(f"  Orgs: {len(ORGS)}")
    print(f"  Employees: {sum(o['num_employees'] for o in ORGS)}")
    print(f"  Questions per employee: {len(QUESTIONS)}")
    print()
    print("Org breakdown:")
    for org in ORGS:
        print(f"  {org['dept_code']:8s} {org['name']:35s} {org['num_employees']:3d} employees")
    print()
    print(f"Output: {output_path}")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic org knowledge Q&A dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/synthetic/raw_responses.jsonl",
        help="Output JSONL file path (default: ./data/synthetic/raw_responses.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    generate_dataset(args.output, args.seed)


if __name__ == "__main__":
    main()
