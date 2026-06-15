import sys
import unittest
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.loader import load_candidates
from pipeline.jd_parser import load_jd_requirements
from pipeline.hard_filters import passes_hard_filters
from pipeline.honeypot_detector import detect_honeypot
from pipeline.ranker import CandidateRanker
import config


class TestPipeline(unittest.TestCase):

    def test_jd_loading(self):
        jd = load_jd_requirements()
        self.assertEqual(jd.title, "Senior AI Engineer")
        self.assertEqual(jd.company, "Redrob AI")
        self.assertTrue(len(config.MUST_HAVE_SKILLS) > 0)

    def test_non_tech_filter(self):
        # Candidate with non-tech current title and non-tech career history should fail
        bad_candidate = {
            "profile": {
                "current_title": "Accountant",
                "years_of_experience": 5
            },
            "career_history": [
                {"title": "Accountant", "duration_months": 24},
                {"title": "HR Manager", "duration_months": 36}
            ],
            "skills": [
                {"name": "Accounting", "proficiency": "expert"}
            ]
        }
        self.assertFalse(passes_hard_filters(bad_candidate))

        # Candidate with non-tech current title but tech history should pass
        good_candidate = {
            "profile": {
                "current_title": "Accountant",
                "years_of_experience": 5
            },
            "career_history": [
                {"title": "Software Engineer", "duration_months": 24},
                {"title": "HR Manager", "duration_months": 36}
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert"}
            ]
        }
        self.assertTrue(passes_hard_filters(good_candidate))

    def test_honeypot_date_math(self):
        # Impossible career timeline
        impossible_candidate = {
            "profile": {
                "current_title": "Software Engineer",
                "years_of_experience": 3
            },
            "career_history": [
                {"title": "Software Engineer", "duration_months": 120}  # 10 years duration for 3 yoe
            ],
            "skills": []
        }
        is_hp, reason = detect_honeypot(impossible_candidate)
        self.assertIn("career_months", reason)

    def test_honeypot_skills(self):
        # Impossible skill durations
        bad_skills = {
            "profile": {
                "current_title": "Software Engineer",
                "years_of_experience": 5
            },
            "career_history": [
                {"title": "Software Engineer", "duration_months": 60}
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 0},
                {"name": "NLP", "proficiency": "advanced", "duration_months": 0},
                {"name": "PyTorch", "proficiency": "expert", "duration_months": 0}
            ]
        }
        is_hp, reason = detect_honeypot(bad_skills)
        self.assertTrue(is_hp)
        self.assertIn("impossible_skills", reason)

    def test_results_validation(self):
        # Valid results list
        valid_results = [
            {"candidate_id": "CAND_001", "rank": 1, "score": 0.95, "reasoning": "Excellent match"},
            {"candidate_id": "CAND_002", "rank": 2, "score": 0.85, "reasoning": "Good match"}
        ]
        errors = CandidateRanker.validate_results(valid_results, strict=False)
        self.assertEqual(len(errors), 0)

        # Invalid results with non-monotonic scores
        invalid_results = [
            {"candidate_id": "CAND_001", "rank": 1, "score": 0.85, "reasoning": "Good match"},
            {"candidate_id": "CAND_002", "rank": 2, "score": 0.95, "reasoning": "Excellent match"}
        ]
        errors = CandidateRanker.validate_results(invalid_results, strict=False)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("non-increasing" in e.lower() for e in errors))


if __name__ == "__main__":
    unittest.main()
