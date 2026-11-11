# Labelling rubric for the support judge

You will see 300 rows. Each row has a question, a one-sentence answer, and one passage. Fill the
`human_label` column with one of three values. Judge only from the passage on the page. Do not use
what you already know about the topic, and do not search.

**supported**: the passage states what the answer says, and what it states answers the question that was asked.

**partial**: the passage is about the right subject and contains part of the answer, or states the answer's
facts but not what the question asked for (for example the question asks when, and the passage describes the
thing without a date). Also use this when the passage supports the answer only if you assume something it
does not say.

**not supported**: the passage does not state the answer, contradicts it, or is about something else, even if it
shares words with the question.

Rules of thumb:
- A matching number or name is not enough. The passage must tie it to the thing the question is about.
- If you would have to guess, choose partial. If you would have to guess between partial and not supported,
  choose not supported.
- Use the optional `notes` column for anything odd. Short is fine.
- Do not look at `judge_labels.csv` while labelling.

Expect about an hour. If the first 30 rows take much longer than 10 minutes, the rubric is unclear: stop and say which rows were hard.
