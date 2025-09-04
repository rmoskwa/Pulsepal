#!/usr/bin/env python3
"""
Regenerate mkdocs.yml and index.md based on existing documentation files.
Comments out missing functions instead of deleting them.
"""

from pathlib import Path
import yaml
import re


def get_existing_docs():
    """Get list of existing documentation files."""
    docs_dir = Path("docs/matlab_api")
    existing = []
    for file in sorted(docs_dir.glob("*.md")):
        if file.name != "index.md":
            existing.append(file.stem)
    return existing


def get_all_expected_functions():
    """Get all functions that should be documented."""
    # These are all the functions that SHOULD exist based on function_index
    all_functions = [
        "EventLibrary",
        "SeqPlot",
        "Sequence",
        "TransformFOV",
        "addBlock",
        "addGradients",
        "addRamps",
        "align",
        "applySoftDelay",
        "applyToBlock",
        "applyToSeq",
        "calcAdcSeg",
        "calcDuration",
        "calcMomentsBtensor",
        "calcPNS",
        "calcRamp",
        "calcRfBandwidth",
        "calcRfCenter",
        "calcRfPower",
        "calculateKspacePP",
        "checkTiming",
        "compressShape",
        "compressShape_mat",
        "conjugate",
        "convert",
        "decompressShape",
        "duration",
        "evalLabels",
        "findBlockByTime",
        "findFlank",
        "flipGradAxis",
        "fromRotMat",
        "getBlock",
        "getDefinition",
        "getSupportedLabels",
        "getSupportedRfUse",
        "install",
        "isOctave",
        "makeAdc",
        "makeAdiabaticPulse",
        "makeArbitraryGrad",
        "makeArbitraryRf",
        "makeBlockPulse",
        "makeDelay",
        "makeDigitalOutputPulse",
        "makeExtendedTrapezoid",
        "makeExtendedTrapezoidArea",
        "makeGaussPulse",
        "makeLabel",
        "makeSLRpulse",
        "makeSincPulse",
        "makeSoftDelay",
        "makeTrapezoid",
        "makeTrigger",
        "md5",
        "melodyToPitchesAndDurations",
        "melodyToScale",
        "modGradAxis",
        "multiply",
        "musicToSequence",
        "normalize",
        "opts",
        "paperPlot",
        "parsemr",
        "plot",
        "pts2waveform",
        "read",
        "readBinary",
        "readasc",
        "registerGradEvent",
        "registerLabelEvent",
        "registerRfEvent",
        "restoreAdditionalShapeSamples",
        "rotate",
        "rotate3D",
        "scaleGrad",
        "setBlock",
        "setDefinition",
        "simRf",
        "sound",
        "splitGradient",
        "splitGradientAt",
        "testReport",
        "toRotMat",
        "traj2grad",
        "transform",
        "version",
        "waveforms_and_times",
        "write",
        "writeBinary",
        "write_v141",
    ]
    return all_functions


def generate_mkdocs_yml(existing_docs):
    """Generate mkdocs.yml with existing files only."""

    # Build the nav structure with only existing files
    nav_functions = []
    for func in sorted(existing_docs):
        nav_functions.append({func: f"matlab_api/{func}.md"})

    config = {
        "site_name": "Pulseq MATLAB API Documentation",
        "site_description": "Complete API reference for Pulseq MATLAB functions",
        "site_url": "https://rmoskwa.github.io/Pulsepal/",
        "theme": {
            "name": "material",
            "palette": {"primary": "blue", "accent": "light blue"},
            "features": [
                "navigation.tabs",
                "navigation.sections",
                "navigation.expand",
                "navigation.top",
                "search.suggest",
                "search.highlight",
                "content.code.copy",
                "content.code.annotate",
            ],
            "font": {"text": "Roboto", "code": "Roboto Mono"},
        },
        "plugins": [{"search": {"lang": "en", "separator": r"[\s\-\.]+"}}],
        "markdown_extensions": [
            "pymdownx.highlight",
            "pymdownx.superfences",
            "pymdownx.tabbed",
            "pymdownx.details",
            "pymdownx.snippets",
            "admonition",
            "tables",
            "toc",
            {"toc": {"permalink": True}},
        ],
        "nav": [
            {"Home": "mdDocs/index.md"},
            {
                "API Reference": [
                    {"Overview": "matlab_api/index.md"},
                    {"Functions": nav_functions},
                ]
            },
        ],
        "extra": {
            "social": [
                {
                    "icon": "fontawesome/brands/github",
                    "link": "https://github.com/pulseq/pulseq",
                }
            ]
        },
    }

    # Write mkdocs.yml
    with open("mkdocs.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("✅ Generated mkdocs.yml")


def update_index_md(existing_docs, all_functions):
    """Update index.md, commenting out missing functions."""

    # Read existing index.md to get descriptions
    index_path = Path("docs/matlab_api/index.md")
    existing_content = index_path.read_text() if index_path.exists() else ""

    # Extract descriptions from existing content
    descriptions = {}
    for line in existing_content.split("\n"):
        match = re.match(r"- \[([^\]]+)\]\([^)]+\) - (.+)", line)
        if match:
            func_name = match.group(1)
            description = match.group(2)
            descriptions[func_name] = description

    # Build new index content
    content = ["# MATLAB API Reference\n"]
    content.append("Complete reference for PulsePal MATLAB functions.\n")

    # Group functions by first letter
    grouped = {}
    for func in sorted(all_functions):
        first_letter = func[0].upper()
        if not first_letter.isalpha():
            first_letter = "#"
        if first_letter not in grouped:
            grouped[first_letter] = []
        grouped[first_letter].append(func)

    # Create sections
    for letter in sorted(grouped.keys()):
        content.append(f"## {letter}\n")

        for func in grouped[letter]:
            description = descriptions.get(func, "")
            if len(description) > 100:
                description = description[:97] + "..."

            if func in existing_docs:
                # Function exists - create normal link
                content.append(f"- [{func}]({func}.md) - {description}")
            else:
                # Function missing - comment it out
                content.append(
                    f"<!-- MISSING: - [{func}]({func}.md) - {description} -->"
                )

        content.append("")

    # Write index.md
    index_path.write_text("\n".join(content))
    print("✅ Updated matlab_api/index.md with commented missing functions")


def main():
    # Get existing documentation files
    existing_docs = get_existing_docs()
    print(f"Found {len(existing_docs)} existing documentation files")

    # Get all expected functions
    all_functions = get_all_expected_functions()

    # Find missing functions
    missing = set(all_functions) - set(existing_docs)
    if missing:
        print(f"Missing documentation for: {', '.join(sorted(missing))}")

    # Generate mkdocs.yml
    generate_mkdocs_yml(existing_docs)

    # Update index.md
    update_index_md(existing_docs, all_functions)

    print("\n✅ Documentation configuration regenerated!")
    print("\nNext steps:")
    print("1. Review the changes")
    print("2. Run: mkdocs build")
    print("3. Deploy: mkdocs gh-deploy")


if __name__ == "__main__":
    main()
