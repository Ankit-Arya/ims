from ike.ingestion.docling_pipeline import structural_heading_aliases, stitch_section_path


def test_split_structural_heading_is_stitched_without_document_specific_vocabulary():
    canonical = {
        "texts": [
            {
                "label": "section_header",
                "text": "CHAPTER IV",
                "level": 1,
                "prov": [{"page_no": 90}],
            },
            {
                "label": "section_header",
                "text": "SPEED AND WORKING OF TRAINS",
                "level": 6,
                "prov": [{"page_no": 90}],
            },
            {
                "label": "section_header",
                "text": "25. General --",
                "level": 3,
                "prov": [{"page_no": 90}],
            },
        ]
    }

    aliases = structural_heading_aliases(canonical)

    assert aliases["chapter iv"] == (
        "CHAPTER IV SPEED AND WORKING OF TRAINS",
        "SPEED AND WORKING OF TRAINS",
    )
    assert stitch_section_path(["CHAPTER IV", "25. General --"], aliases) == [
        "CHAPTER IV SPEED AND WORKING OF TRAINS",
        "25. General --",
    ]


def test_existing_title_component_is_not_duplicated():
    canonical = {
        "texts": [
            {
                "label": "section_header",
                "text": "PART 2",
                "level": 1,
                "prov": [{"page_no": 10}],
            },
            {
                "label": "section_header",
                "text": "Operating Requirements",
                "level": 4,
                "prov": [{"page_no": 10}],
            },
        ]
    }

    aliases = structural_heading_aliases(canonical)

    assert stitch_section_path(
        ["PART 2", "Operating Requirements"],
        aliases,
    ) == ["PART 2 Operating Requirements"]


def test_rule_heading_is_not_mistaken_for_structural_title():
    canonical = {
        "texts": [
            {
                "label": "section_header",
                "text": "CHAPTER VII",
                "level": 1,
                "prov": [{"page_no": 25}],
            },
            {
                "label": "section_header",
                "text": "48. Report of accident",
                "level": 3,
                "prov": [{"page_no": 25}],
            },
        ]
    }

    assert structural_heading_aliases(canonical) == {}
