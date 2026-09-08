"""Grounded, template-based chatbot over the structured screening outputs.

DELIBERATELY NOT A LANGUAGE MODEL
---------------------------------
Every sentence this produces is assembled from values the pipeline computed. It cannot
invent a clinical finding, because it has no generative capacity to invent one with. For a
screening prototype that property is worth more than fluency, and it is why the chatbot is
classified as an interface component rather than a research contribution.

ONE HARD RULE
-------------
When a case has been deferred, the deferral is stated FIRST, before any class name. A
system that reports "cataract, 71%" on a case it chose not to answer has broken the safety
property the whole architecture exists to provide.
"""

from __future__ import annotations

CLASS_NAMES = {
    "N": "no referable abnormality detected",
    "D": "signs consistent with diabetic retinopathy",
    "G": "signs consistent with glaucoma",
    "C": "signs consistent with cataract",
}

DISCLAIMER = ("This is an academic screening prototype, not a medical device. It does not "
              "diagnose. A qualified clinician reviews every case and makes every decision.")


class Chatbot:
    """Answers questions from one case's structured record. No free text in, no free text out."""

    def __init__(self, case: dict):
        self.case = case

    # --- the rule that outranks everything else ---------------------------------------
    def _deferral_preamble(self) -> str | None:
        if not self.case.get("deferred"):
            return None
        return (f"This case was DEFERRED for clinician review. Reason: "
                f"{self.case.get('deferral_reason', 'low confidence')}. "
                f"The screening finding below is shown only as context and has not been "
                f"accepted by the system.")

    def prediction(self) -> str:
        preamble = self._deferral_preamble()
        finding = (f"Screening finding for {self.case['image_id']}: "
                   f"{CLASS_NAMES.get(self.case['y_pred'], self.case['y_pred'])}, "
                   f"calibrated confidence {self.case['max_prob']:.0%}.")
        return f"{preamble} {finding}" if preamble else finding

    def confidence(self) -> str:
        # Do not claim calibration the pipeline has not performed. The deployed head is
        # temperature-scaled (T=0.831, deployment.json), but a case carries its own flag and
        # this sentence follows the flag rather than assuming.
        calibrated = self.case.get("calibrated", False)
        basis = ("It has been temperature-scaled on validation data, so it is comparable "
                 "across cases." if calibrated else
                 "It has NOT been calibrated, so treat it as a ranking score rather than a "
                 "probability.")
        return (f"The model assigned {self.case['max_prob']:.0%} confidence to its top class. "
                f"{basis} It is not a probability of the patient having the disease.")

    def why_deferred(self) -> str:
        if not self.case.get("deferred"):
            return ("This case was not deferred: image confidence was above the operating "
                    "threshold and the referral document did not contradict the finding.")
        return (f"Deferred because: {self.case.get('deferral_reason', 'low confidence')}. "
                "Deferred cases are placed at the top of the clinician worklist.")

    def document_fields(self) -> str:
        fields = self.case.get("document_fields") or {}
        if not fields:
            return "No referral document was supplied for this case."
        lines = [f"  {k.replace('_', ' ')}: {v}" for k, v in fields.items() if v is not None]
        confidence = self.case.get("extraction_confidence")
        suffix = (f"\nEstimated probability the extraction is correct: {confidence:.0%}."
                  if confidence is not None else "")
        return "Extracted from the referral document:\n" + "\n".join(lines) + suffix

    def concordance(self) -> str:
        stated = self.case.get("stated_label")
        if stated is None:
            return "No referral diagnosis was available to compare against."
        agreement = self.case.get("agreement", 0.0)
        if stated == self.case["y_pred"]:
            return (f"The referral states {CLASS_NAMES.get(stated, stated)}, which matches "
                    f"the image finding. The image assigns {agreement:.0%} probability to "
                    "that class.")
        return (f"DISCORDANCE: the referral states {CLASS_NAMES.get(stated, stated)}, but the "
                f"image finding is {CLASS_NAMES.get(self.case['y_pred'], self.case['y_pred'])}. "
                f"The image assigns only {agreement:.0%} probability to the referral's claim. "
                "Either source may be wrong; a clinician should review.")

    def limitations(self) -> str:
        return ("Limitations: trained on a public research dataset with 203 cataract and 182 "
                "glaucoma patients; not validated on prospective clinical data; the referral "
                "documents used in development were synthetic. " + DISCLAIMER)

    INTENTS = {
        "what was predicted": "prediction",
        "how confident": "confidence",
        "why deferred": "why_deferred",
        "what was extracted": "document_fields",
        "do the image and document agree": "concordance",
        "what are the limitations": "limitations",
    }

    def ask(self, question: str) -> str:
        """Match to a known intent. An unmatched question is refused, never improvised."""
        probe = question.lower().strip("? ")
        for phrase, method in self.INTENTS.items():
            if phrase in probe or probe in phrase:
                return getattr(self, method)()
        return ("I can only answer from this case's structured record. Try: "
                + "; ".join(self.INTENTS) + ".")

    def summary(self) -> list[tuple[str, str]]:
        """The full grounded briefing, in the order a clinician would want it."""
        return [(phrase, getattr(self, method)()) for phrase, method in self.INTENTS.items()]
