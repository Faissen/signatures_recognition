from signature_utils import compare_all_signatures

tests = [
    "../signatures_to_test/signature_to_check.png",
    "../signatures_to_test/test_david_bradford.png"
]

for img in tests:
    print(f"\nResultados para: {img}")
    result = compare_all_signatures(img)

    for name, score in result["top_3_matches"]:
        print(f"{name}: match = {round(score, 1)}%")
