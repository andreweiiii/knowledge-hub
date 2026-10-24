"""Sample documents loaded into an empty knowledge base so the app is useful on first run."""

SEED_DOCUMENTS = [
    {
        "name": "Employee Travel & Expense Policy",
        "filename": "employee-travel-expense-policy.pdf",
        "category": "Policies",
        "content": """
Travel expense claims must include an approved travel request, compliant transportation and accommodation invoices, travel proof such as a boarding pass or train ticket, and an itemized expense report. Employees should submit the claim within 10 business days after returning. Claims submitted after the deadline require a written explanation from the employee's manager.
        """,
    },
    {
        "name": "New Employee Onboarding Handbook 2026",
        "filename": "onboarding-handbook-2026.pdf",
        "category": "Policies",
        "content": """
During the first week, a new employee should complete account activation, equipment collection, information security training, role documentation, and a 30-day goal review with their manager. The manager should schedule a one-to-one meeting before the end of the first week to confirm access, expectations, and the learning plan.
        """,
    },
    {
        "name": "V3.2 Product Release Notes",
        "filename": "product-release-v3.2.docx",
        "category": "Product",
        "content": """
Before releasing V3.2, the product owner must confirm the release scope, QA must complete regression testing, and the technical owner must document the rollout, rollback, and monitoring plans. Customer support and operations must receive the FAQ and external announcement before release. Production approvals should be completed at least one business day before the release time.
        """,
    },
    {
        "name": "Engineering Collaboration & Git Guide",
        "filename": "engineering-git-guide.md",
        "category": "Engineering",
        "content": """
Feature branches should be created from main and named with the ticket identifier. Pull requests need one code owner approval and a passing CI check before merging. Production releases use a tagged release branch and must include a rollback plan. Production access is requested through an access ticket and is logged for audit.
        """,
    },
]
