# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
generate_data.py
-----------------
Builds a knowledge base + QA set + a bank of "generated" RAG answers with
controlled, ground-truth-labeled hallucinations.

Domain: astronomy and space physics. Every document below states a real,
verifiable fact, phrased in original wording (not copied from any source)
so there are no copyright concerns. Because we author the hallucinated
variants ourselves, we get *exact* claim-level ground truth to evaluate
the faithfulness detector against — something you can't easily get with a
real LLM's free-form output.

Hallucination types injected (mirrors Section 7.2.3 / 7.5.4 of the survey
"A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems"):
  - grounded       : answer fully supported by the retrieved context
  - numeric        : a number in the answer is silently changed
  - fabrication    : an extra, invented sentence is appended
  - contradiction  : a qualitative fact is flipped (e.g. fastest -> slowest)
"""
import json
import os

CORPUS = [
    ("doc_01", "The speed of light in a vacuum is 299,792 kilometers per second, a physical constant used to define the meter since 1983."),
    ("doc_02", "The Chandrasekhar limit, approximately 1.4 solar masses, defines the maximum mass at which a white dwarf star can remain stable against gravitational collapse."),
    ("doc_03", "Current cosmological measurements place the age of the universe at approximately 13.8 billion years, based on observations of the cosmic microwave background."),
    ("doc_04", "The Hubble constant, which describes the rate of cosmic expansion, is measured at approximately 70 kilometers per second per megaparsec."),
    ("doc_05", "Sagittarius A*, the supermassive black hole at the center of the Milky Way, has an estimated mass of 4.3 million solar masses and lies about 26,000 light-years from Earth."),
    ("doc_06", "Voyager 1 was launched in September 1977 and crossed into interstellar space in August 2012, making it the most distant human-made object."),
    ("doc_07", "The James Webb Space Telescope launched in December 2021 and operates near the Sun-Earth L2 point, roughly 1.5 million kilometers from Earth."),
    ("doc_08", "The Sun contains about 99.86 percent of the total mass in the solar system and has a radius of approximately 696,000 kilometers."),
    ("doc_09", "Earth completes one orbit around the Sun in 365.25 days at an average distance of 149.6 million kilometers, defined as one astronomical unit."),
    ("doc_10", "Proxima Centauri, the nearest known star to the Sun, lies approximately 4.24 light-years away in the Alpha Centauri system."),
    ("doc_11", "The Milky Way galaxy spans roughly 100,000 light-years in diameter and contains an estimated 200 billion stars."),
    ("doc_12", "Neutron star material is so dense that a single teaspoon would have a mass of roughly 10 million tons."),
    ("doc_13", "The cosmic microwave background radiation has a temperature of approximately 2.725 kelvin and was first detected in 1965 by Arno Penzias and Robert Wilson."),
    ("doc_14", "Albert Einstein published his general theory of relativity in 1915, describing gravity as the curvature of spacetime."),
    ("doc_15", "51 Pegasi b, discovered in 1995, was the first exoplanet found orbiting a Sun-like star."),
    ("doc_16", "The International Space Station orbits Earth at an altitude of about 400 kilometers, completing one full orbit approximately every 90 minutes."),
    ("doc_17", "Halley's Comet has an orbital period of about 76 years and was last visible from Earth in 1986."),
    ("doc_18", "NASA's Kepler space telescope, active from 2009 to 2018, confirmed the discovery of over 2,600 exoplanets."),
    ("doc_19", "One light-year, the distance light travels in a vacuum in one year, is equal to about 9.46 trillion kilometers."),
    ("doc_20", "Mars has two small moons, Phobos and Deimos, both discovered by astronomer Asaph Hall in 1877."),
    ("doc_21", "Jupiter is the largest planet in the solar system, with a mass about 318 times that of Earth."),
    ("doc_22", "In 2019, the Event Horizon Telescope produced the first direct image of a black hole, M87*, estimated to have a mass of 6.5 billion solar masses."),
    ("doc_23", "The pulsar PSR J1748-2446ad is the fastest-spinning neutron star known, rotating approximately 716 times per second."),
    ("doc_24", "The observable universe has a diameter of about 93 billion light-years, larger than its age in light-years due to the expansion of space."),
    ("doc_25", "The escape velocity needed to leave Earth's gravitational field from the surface is approximately 11.2 kilometers per second."),
]

# Each entry: question, source doc id, and the four answer variants.
# claims: list of (text, gold_label) where gold_label in {supported, unsupported, contradicted}
QA_ITEMS = [
    dict(
        qid="q01", doc_id="doc_01",
        question="What is the speed of light in a vacuum, and why is that value significant?",
        variants=dict(
            grounded=[("The speed of light in a vacuum is 299,792 kilometers per second, and this constant has been used to define the meter since 1983.", "supported")],
            numeric=[("The speed of light in a vacuum is 150,000 kilometers per second, and this constant has been used to define the meter since 1983.", "contradicted")],
            fabrication=[("The speed of light in a vacuum is 299,792 kilometers per second, and this constant has been used to define the meter since 1983.", "supported"),
                         ("This value was first measured precisely by Isaac Newton using a rotating mirror apparatus.", "unsupported")],
            contradiction=[("The speed of light in a vacuum varies depending on the observer's velocity and has never been used to define any unit.", "contradicted")],
        ),
    ),
    dict(
        qid="q02", doc_id="doc_02",
        question="What is the Chandrasekhar limit?",
        variants=dict(
            grounded=[("The Chandrasekhar limit is approximately 1.4 solar masses, the maximum mass at which a white dwarf star can remain stable against gravitational collapse.", "supported")],
            numeric=[("The Chandrasekhar limit is approximately 8.2 solar masses, the maximum mass at which a white dwarf star can remain stable against gravitational collapse.", "contradicted")],
            fabrication=[("The Chandrasekhar limit is approximately 1.4 solar masses, the maximum mass at which a white dwarf star can remain stable against gravitational collapse.", "supported"),
                         ("The limit was named after a telescope launched by NASA in 1999.", "unsupported")],
            contradiction=[("The Chandrasekhar limit describes the minimum mass a star needs to ignite nuclear fusion, not a stability threshold for white dwarfs.", "contradicted")],
        ),
    ),
    dict(
        qid="q03", doc_id="doc_03",
        question="How old is the universe according to current cosmological measurements?",
        variants=dict(
            grounded=[("Current measurements place the age of the universe at approximately 13.8 billion years, based on cosmic microwave background observations.", "supported")],
            numeric=[("Current measurements place the age of the universe at approximately 4.6 billion years, based on cosmic microwave background observations.", "contradicted")],
            fabrication=[("Current measurements place the age of the universe at approximately 13.8 billion years, based on cosmic microwave background observations.", "supported"),
                         ("This estimate was first confirmed by the Apollo 11 lunar samples.", "unsupported")],
            contradiction=[("The age of the universe is not known and cosmic microwave background observations provide no constraint on it.", "contradicted")],
        ),
    ),
    dict(
        qid="q04", doc_id="doc_04",
        question="What is the Hubble constant?",
        variants=dict(
            grounded=[("The Hubble constant, describing the rate of cosmic expansion, is measured at approximately 70 kilometers per second per megaparsec.", "supported")],
            numeric=[("The Hubble constant, describing the rate of cosmic expansion, is measured at approximately 210 kilometers per second per megaparsec.", "contradicted")],
            fabrication=[("The Hubble constant, describing the rate of cosmic expansion, is measured at approximately 70 kilometers per second per megaparsec.", "supported"),
                         ("It was first calculated using data from the Voyager 1 probe.", "unsupported")],
            contradiction=[("The Hubble constant shows that the universe's expansion is slowing down and will eventually reverse into a contraction.", "contradicted")],
        ),
    ),
    dict(
        qid="q05", doc_id="doc_05",
        question="What is Sagittarius A* and how far is it from Earth?",
        variants=dict(
            grounded=[("Sagittarius A*, the supermassive black hole at the Milky Way's center, has an estimated mass of 4.3 million solar masses and lies about 26,000 light-years from Earth.", "supported")],
            numeric=[("Sagittarius A*, the supermassive black hole at the Milky Way's center, has an estimated mass of 4.3 million solar masses and lies about 900 light-years from Earth.", "contradicted")],
            fabrication=[("Sagittarius A*, the supermassive black hole at the Milky Way's center, has an estimated mass of 4.3 million solar masses and lies about 26,000 light-years from Earth.", "supported"),
                         ("It was first photographed directly in 1998 by the Hubble Space Telescope.", "unsupported")],
            contradiction=[("Sagittarius A* is located at the outer edge of the Milky Way, far from the galactic center, about 26,000 light-years from Earth.", "contradicted")],
        ),
    ),
    dict(
        qid="q06", doc_id="doc_06",
        question="When did Voyager 1 enter interstellar space?",
        variants=dict(
            grounded=[("Voyager 1 was launched in September 1977 and crossed into interstellar space in August 2012, making it the most distant human-made object.", "supported")],
            numeric=[("Voyager 1 was launched in September 1977 and crossed into interstellar space in August 1989, making it the most distant human-made object.", "contradicted")],
            fabrication=[("Voyager 1 was launched in September 1977 and crossed into interstellar space in August 2012, making it the most distant human-made object.", "supported"),
                         ("Voyager 1 carries a working camera that still streams live images of interstellar space today.", "unsupported")],
            contradiction=[("Voyager 1 remains within the solar system and has never reached interstellar space.", "contradicted")],
        ),
    ),
    dict(
        qid="q07", doc_id="doc_07",
        question="Where does the James Webb Space Telescope operate?",
        variants=dict(
            grounded=[("The James Webb Space Telescope, launched in December 2021, operates near the Sun-Earth L2 point, roughly 1.5 million kilometers from Earth.", "supported")],
            numeric=[("The James Webb Space Telescope, launched in December 2021, operates near the Sun-Earth L2 point, roughly 42,000 kilometers from Earth.", "contradicted")],
            fabrication=[("The James Webb Space Telescope, launched in December 2021, operates near the Sun-Earth L2 point, roughly 1.5 million kilometers from Earth.", "supported"),
                         ("It is serviced by astronauts every two years during dedicated shuttle missions.", "unsupported")],
            contradiction=[("The James Webb Space Telescope orbits directly above Earth's surface at low altitude, close enough for astronauts to service it.", "contradicted")],
        ),
    ),
    dict(
        qid="q08", doc_id="doc_08",
        question="What share of the solar system's mass does the Sun hold, and how large is it?",
        variants=dict(
            grounded=[("The Sun contains about 99.86 percent of the total mass in the solar system and has a radius of approximately 696,000 kilometers.", "supported")],
            numeric=[("The Sun contains about 54 percent of the total mass in the solar system and has a radius of approximately 696,000 kilometers.", "contradicted")],
            fabrication=[("The Sun contains about 99.86 percent of the total mass in the solar system and has a radius of approximately 696,000 kilometers.", "supported"),
                         ("The Sun's core will run out of hydrogen fuel within the next 50 years.", "unsupported")],
            contradiction=[("The Sun holds only a small minority of the solar system's total mass, with most of it concentrated in the outer planets.", "contradicted")],
        ),
    ),
    dict(
        qid="q09", doc_id="doc_09",
        question="How long does it take Earth to orbit the Sun, and what defines an astronomical unit?",
        variants=dict(
            grounded=[("Earth completes one orbit around the Sun in 365.25 days at an average distance of 149.6 million kilometers, which defines one astronomical unit.", "supported")],
            numeric=[("Earth completes one orbit around the Sun in 365.25 days at an average distance of 12.7 million kilometers, which defines one astronomical unit.", "contradicted")],
            fabrication=[("Earth completes one orbit around the Sun in 365.25 days at an average distance of 149.6 million kilometers, which defines one astronomical unit.", "supported"),
                         ("This distance was first measured precisely using radar ranging of Venus in the 1960s.", "unsupported")],
            contradiction=[("Earth's distance from the Sun changes so unpredictably that no standard astronomical unit could ever be defined from it.", "contradicted")],
        ),
    ),
    dict(
        qid="q10", doc_id="doc_10",
        question="How far away is Proxima Centauri?",
        variants=dict(
            grounded=[("Proxima Centauri, the nearest known star to the Sun, lies approximately 4.24 light-years away in the Alpha Centauri system.", "supported")],
            numeric=[("Proxima Centauri, the nearest known star to the Sun, lies approximately 47 light-years away in the Alpha Centauri system.", "contradicted")],
            fabrication=[("Proxima Centauri, the nearest known star to the Sun, lies approximately 4.24 light-years away in the Alpha Centauri system.", "supported"),
                         ("A crewed mission to Proxima Centauri is currently scheduled to launch in 2030.", "unsupported")],
            contradiction=[("Proxima Centauri is the farthest known star from the Sun, lying approximately 4.24 light-years away.", "contradicted")],
        ),
    ),
    dict(
        qid="q11", doc_id="doc_11",
        question="How large is the Milky Way and how many stars does it contain?",
        variants=dict(
            grounded=[("The Milky Way galaxy spans roughly 100,000 light-years in diameter and contains an estimated 200 billion stars.", "supported")],
            numeric=[("The Milky Way galaxy spans roughly 100,000 light-years in diameter and contains an estimated 3 million stars.", "contradicted")],
            fabrication=[("The Milky Way galaxy spans roughly 100,000 light-years in diameter and contains an estimated 200 billion stars.", "supported"),
                         ("At its exact center lies a dense cluster of habitable exoplanets.", "unsupported")],
            contradiction=[("The Milky Way is one of the smallest known galaxies, spanning roughly 100,000 light-years in diameter.", "contradicted")],
        ),
    ),
    dict(
        qid="q12", doc_id="doc_12",
        question="How dense is neutron star material?",
        variants=dict(
            grounded=[("Neutron star material is so dense that a single teaspoon would have a mass of roughly 10 million tons.", "supported")],
            numeric=[("Neutron star material is so dense that a single teaspoon would have a mass of roughly 40 tons.", "contradicted")],
            fabrication=[("Neutron star material is so dense that a single teaspoon would have a mass of roughly 10 million tons.", "supported"),
                         ("This density was first replicated artificially in a laboratory in 2015.", "unsupported")],
            contradiction=[("Neutron star material is among the least dense substances known, far less dense than ordinary water.", "contradicted")],
        ),
    ),
    dict(
        qid="q13", doc_id="doc_13",
        question="What is the temperature of the cosmic microwave background, and who discovered it?",
        variants=dict(
            grounded=[("The cosmic microwave background has a temperature of approximately 2.725 kelvin and was first detected in 1965 by Arno Penzias and Robert Wilson.", "supported")],
            numeric=[("The cosmic microwave background has a temperature of approximately 96 kelvin and was first detected in 1965 by Arno Penzias and Robert Wilson.", "contradicted")],
            fabrication=[("The cosmic microwave background has a temperature of approximately 2.725 kelvin and was first detected in 1965 by Arno Penzias and Robert Wilson.", "supported"),
                         ("Penzias and Wilson made the discovery while testing a satellite communications antenna, and were initially trying to eliminate what they thought was noise.", "unsupported")],
            contradiction=[("The cosmic microwave background was never detected experimentally and remains a purely theoretical prediction.", "contradicted")],
        ),
    ),
    dict(
        qid="q14", doc_id="doc_14",
        question="When did Einstein publish his general theory of relativity?",
        variants=dict(
            grounded=[("Albert Einstein published his general theory of relativity in 1915, describing gravity as the curvature of spacetime.", "supported")],
            numeric=[("Albert Einstein published his general theory of relativity in 1953, describing gravity as the curvature of spacetime.", "contradicted")],
            fabrication=[("Albert Einstein published his general theory of relativity in 1915, describing gravity as the curvature of spacetime.", "supported"),
                         ("Einstein received the Nobel Prize specifically for this theory the following year.", "unsupported")],
            contradiction=[("Einstein's general theory of relativity describes gravity as a force transmitted instantaneously through empty space, not as spacetime curvature.", "contradicted")],
        ),
    ),
    dict(
        qid="q15", doc_id="doc_15",
        question="What was the first exoplanet discovered around a Sun-like star?",
        variants=dict(
            grounded=[("51 Pegasi b, discovered in 1995, was the first exoplanet found orbiting a Sun-like star.", "supported")],
            numeric=[("51 Pegasi b, discovered in 1961, was the first exoplanet found orbiting a Sun-like star.", "contradicted")],
            fabrication=[("51 Pegasi b, discovered in 1995, was the first exoplanet found orbiting a Sun-like star.", "supported"),
                         ("51 Pegasi b has since been confirmed to have liquid water on its surface.", "unsupported")],
            contradiction=[("51 Pegasi b was the last exoplanet ever discovered orbiting a Sun-like star, closing out the search in 1995.", "contradicted")],
        ),
    ),
    dict(
        qid="q16", doc_id="doc_16",
        question="At what altitude does the International Space Station orbit, and how fast does it circle Earth?",
        variants=dict(
            grounded=[("The International Space Station orbits Earth at an altitude of about 400 kilometers, completing one full orbit approximately every 90 minutes.", "supported")],
            numeric=[("The International Space Station orbits Earth at an altitude of about 35,000 kilometers, completing one full orbit approximately every 90 minutes.", "contradicted")],
            fabrication=[("The International Space Station orbits Earth at an altitude of about 400 kilometers, completing one full orbit approximately every 90 minutes.", "supported"),
                         ("The station has been continuously occupied by an international crew since 1985.", "unsupported")],
            contradiction=[("The International Space Station is a stationary outpost that does not orbit Earth at all.", "contradicted")],
        ),
    ),
    dict(
        qid="q17", doc_id="doc_17",
        question="How often does Halley's Comet appear, and when was it last seen?",
        variants=dict(
            grounded=[("Halley's Comet has an orbital period of about 76 years and was last visible from Earth in 1986.", "supported")],
            numeric=[("Halley's Comet has an orbital period of about 12 years and was last visible from Earth in 1986.", "contradicted")],
            fabrication=[("Halley's Comet has an orbital period of about 76 years and was last visible from Earth in 1986.", "supported"),
                         ("Halley's Comet is expected to collide with Mars during its next close approach.", "unsupported")],
            contradiction=[("Halley's Comet follows no predictable orbital period and appears at random, unpredictable intervals.", "contradicted")],
        ),
    ),
    dict(
        qid="q18", doc_id="doc_18",
        question="How many exoplanets did NASA's Kepler telescope confirm?",
        variants=dict(
            grounded=[("NASA's Kepler space telescope, active from 2009 to 2018, confirmed the discovery of over 2,600 exoplanets.", "supported")],
            numeric=[("NASA's Kepler space telescope, active from 2009 to 2018, confirmed the discovery of over 40 exoplanets.", "contradicted")],
            fabrication=[("NASA's Kepler space telescope, active from 2009 to 2018, confirmed the discovery of over 2,600 exoplanets.", "supported"),
                         ("Kepler was decommissioned after colliding with a fragment of space debris.", "unsupported")],
            contradiction=[("NASA's Kepler space telescope failed shortly after launch and confirmed no exoplanet discoveries at all.", "contradicted")],
        ),
    ),
    dict(
        qid="q19", doc_id="doc_19",
        question="How many kilometers are in a light-year?",
        variants=dict(
            grounded=[("One light-year, the distance light travels in a vacuum in one year, is equal to about 9.46 trillion kilometers.", "supported")],
            numeric=[("One light-year, the distance light travels in a vacuum in one year, is equal to about 384,000 kilometers.", "contradicted")],
            fabrication=[("One light-year, the distance light travels in a vacuum in one year, is equal to about 9.46 trillion kilometers.", "supported"),
                         ("The term was first coined by NASA engineers during the Apollo program.", "unsupported")],
            contradiction=[("A light-year is a measure of time, not distance, describing how long light takes to travel between two points.", "contradicted")],
        ),
    ),
    dict(
        qid="q20", doc_id="doc_20",
        question="What moons does Mars have, and who discovered them?",
        variants=dict(
            grounded=[("Mars has two small moons, Phobos and Deimos, both discovered by astronomer Asaph Hall in 1877.", "supported")],
            numeric=[("Mars has two small moons, Phobos and Deimos, both discovered by astronomer Asaph Hall in 1954.", "contradicted")],
            fabrication=[("Mars has two small moons, Phobos and Deimos, both discovered by astronomer Asaph Hall in 1877.", "supported"),
                         ("Phobos is gradually spiraling outward and will eventually escape Mars's orbit entirely.", "unsupported")],
            contradiction=[("Mars has no natural moons and has never been observed to have any.", "contradicted")],
        ),
    ),
    dict(
        qid="q21", doc_id="doc_21",
        question="Which is the largest planet in the solar system, and how massive is it?",
        variants=dict(
            grounded=[("Jupiter is the largest planet in the solar system, with a mass about 318 times that of Earth.", "supported")],
            numeric=[("Jupiter is the largest planet in the solar system, with a mass about 9 times that of Earth.", "contradicted")],
            fabrication=[("Jupiter is the largest planet in the solar system, with a mass about 318 times that of Earth.", "supported"),
                         ("Jupiter has a solid rocky surface that future missions plan to land crewed vehicles on.", "unsupported")],
            contradiction=[("Jupiter is the smallest planet in the solar system, with a mass about 318 times that of Earth.", "contradicted")],
        ),
    ),
    dict(
        qid="q22", doc_id="doc_22",
        question="When was the first direct image of a black hole taken, and how massive is that black hole?",
        variants=dict(
            grounded=[("In 2019, the Event Horizon Telescope produced the first direct image of a black hole, M87*, estimated to have a mass of 6.5 billion solar masses.", "supported")],
            numeric=[("In 2019, the Event Horizon Telescope produced the first direct image of a black hole, M87*, estimated to have a mass of 12 solar masses.", "contradicted")],
            fabrication=[("In 2019, the Event Horizon Telescope produced the first direct image of a black hole, M87*, estimated to have a mass of 6.5 billion solar masses.", "supported"),
                         ("The image was captured using a single ground-based optical telescope in Chile.", "unsupported")],
            contradiction=[("No image of a black hole has ever been produced, as their nature makes direct imaging theoretically impossible.", "contradicted")],
        ),
    ),
    dict(
        qid="q23", doc_id="doc_23",
        question="What is the fastest-spinning neutron star known, and how fast does it rotate?",
        variants=dict(
            grounded=[("The pulsar PSR J1748-2446ad is the fastest-spinning neutron star known, rotating approximately 716 times per second.", "supported")],
            numeric=[("The pulsar PSR J1748-2446ad is the fastest-spinning neutron star known, rotating approximately 3 times per second.", "contradicted")],
            fabrication=[("The pulsar PSR J1748-2446ad is the fastest-spinning neutron star known, rotating approximately 716 times per second.", "supported"),
                         ("It was discovered by amateur astronomers using a backyard telescope in 2010.", "unsupported")],
            contradiction=[("The pulsar PSR J1748-2446ad is the slowest-spinning neutron star known, rotating approximately 716 times per second.", "contradicted")],
        ),
    ),
    dict(
        qid="q24", doc_id="doc_24",
        question="How large is the observable universe compared to its age?",
        variants=dict(
            grounded=[("The observable universe has a diameter of about 93 billion light-years, larger than its age in light-years due to the expansion of space.", "supported")],
            numeric=[("The observable universe has a diameter of about 13.8 light-years, larger than its age in light-years due to the expansion of space.", "contradicted")],
            fabrication=[("The observable universe has a diameter of about 93 billion light-years, larger than its age in light-years due to the expansion of space.", "supported"),
                         ("Astronomers expect to observe its true edge directly once the next generation of telescopes launches in 2030.", "unsupported")],
            contradiction=[("The observable universe is smaller than its age in light-years would suggest, since space does not expand.", "contradicted")],
        ),
    ),
    dict(
        qid="q25", doc_id="doc_25",
        question="What is Earth's escape velocity?",
        variants=dict(
            grounded=[("The escape velocity needed to leave Earth's gravitational field from the surface is approximately 11.2 kilometers per second.", "supported")],
            numeric=[("The escape velocity needed to leave Earth's gravitational field from the surface is approximately 0.9 kilometers per second.", "contradicted")],
            fabrication=[("The escape velocity needed to leave Earth's gravitational field from the surface is approximately 11.2 kilometers per second.", "supported"),
                         ("This figure was first calculated experimentally by test-firing early V2 rockets.", "unsupported")],
            contradiction=[("Earth has no meaningful escape velocity because its gravitational field does not extend beyond the upper atmosphere.", "contradicted")],
        ),
    ),
]


def build():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    corpus = [{"doc_id": d, "text": t} for d, t in CORPUS]
    with open(os.path.join(data_dir, "corpus.json"), "w") as f:
        json.dump(corpus, f, indent=2)

    qa = []
    answers = []
    for item in QA_ITEMS:
        qa.append({"qid": item["qid"], "question": item["question"], "gold_doc_id": item["doc_id"]})
        for variant_name, claims in item["variants"].items():
            answers.append({
                "qid": item["qid"],
                "question": item["question"],
                "answer_id": f"{item['qid']}_{variant_name}",
                "answer_type": variant_name,
                "answer_text": " ".join(c[0] for c in claims),
                "claims": [{"text": c[0], "gold_label": c[1]} for c in claims],
            })

    with open(os.path.join(data_dir, "qa_dataset.json"), "w") as f:
        json.dump(qa, f, indent=2)
    with open(os.path.join(data_dir, "generated_answers.json"), "w") as f:
        json.dump(answers, f, indent=2)

    print(f"Wrote {len(corpus)} corpus docs, {len(qa)} QA items, {len(answers)} generated answers "
          f"({len(answers) // len(qa)} variants each) to {data_dir}")


if __name__ == "__main__":
    build()
