# MultiSourceQA

Small multi-source company dataset with four inputs:
- `company.jsonl`
- `customers.jsonl`
- `calendar.jsonl`
- `slack.jsonl`

The dataset is meant to simulate a small B2B software company viewed through several internal systems.
Each file holds a different kind of information about the same organization, so questions often require combining multiple sources.

## Sources

`company.jsonl`

Reference-style internal documents.
These records contain slower-moving company knowledge such as policies, org structure, office details, roadmap notes, security process, board schedule, training catalog, and planning deadlines.
The important fields are:
- `doc_type`: broad category like policy, product, finance, or people
- `title`: document title
- `last_updated`: when the document was last updated
- `text`: the actual policy or reference content

`customers.jsonl`

CRM-style customer account snapshots.
These records describe the current state of each customer relationship rather than a point-in-time event.
They include ownership, contract size, renewal timing, contacts, open opportunities, and short account notes.
The important fields are:
- `customer_name` and `customer_id`: account identity
- `account_owner` and `csm`: internal commercial owners
- `segment`, `region`, `plan`, `seats`, `annual_value_eur`: commercial profile
- `renewal_date`: next renewal milestone
- `billing_contact`, `procurement_contact`, `security_contact`: customer-side contacts
- `open_opportunities` and `notes`: current account context

`calendar.jsonl`

Future scheduled events.
These are concrete meetings, maintenance windows, training sessions, internal reviews, and office activities that happen after the dataset's effective `now`.
The important fields are:
- `start` and `end`: scheduled time range
- `title`: event name
- `participants`: people or groups expected to attend
- `location`: where the event happens
- `related_entities`: customer, office, or initiative linked to the event
- `notes`: short event context

`slack.jsonl`

Short operational updates and internal coordination messages. 
These records are the most conversational source and capture what the company is actively discussing over time, including reminders, product updates, support incidents, legal/commercial blockers, and planning notes.
The important fields are:
- `timestamp`: when the message was sent
- `channel`: team or functional channel
- `sender`: employee who posted the message
- `text`: message body
- `thread_id`: thread identifier
- `entities`: key people, customers, offices, or initiatives mentioned in the message

## Audit Timeline

Use the audit script to anchor `now` to the latest Slack message and check that:
- company docs are not newer than `now`
- calendar events are strictly after `now`

```bash
python data/multisourceqa/timeline_audit.py
```
