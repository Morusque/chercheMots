<?php

ini_set('memory_limit', '512M');
ini_set('max_execution_time', 300);

header('Content-Type: application/json');

$reader = new XMLReader();
if (!$reader->open('words_database.xml')) {
    echo json_encode(["error" => "Failed to open XML database."]);
    exit;
}

$startsWith = $_POST["startsWith"] ?? '';
$natures = isset($_POST['natures']) ? explode(',', $_POST['natures']) : [];
$syllablesSelected = isset($_POST['syllables']) ? explode(',', $_POST['syllables']) : [];
$rhymeWith = strtolower($_POST['rhymeWith']);
$syllableSlider = (int)($_POST['syllableSlider'] ?? 0); // Minimum number of syllables to match for rhymes
$semanticallyClose = $_POST["semanticallyClose"];
$frequencyRanges = explode(',', $_POST['frequencies'] ?? '');
$selectedGenre = $_POST['genre'] ?? ''; // Example: 'm', 'f', or ''
$selectedNombre = $_POST['nombre'] ?? ''; // Example: 's', 'p', or ''
$reverseVectors = $_POST['reverseVectors'] ?? false; // Reverse the vectors for cosine similarity

$results = [];

// Find the pronunciation for the "rhymeWith" word
$rhymePronunciations = [];
// Find the target vectors
$targetVector = null;
if ($rhymeWith || $semanticallyClose) {
	
    while ($reader->read()) {
        if ($reader->nodeType == XMLReader::ELEMENT && $reader->name == 'word') {
            $doc = new DOMDocument();
            $node = simplexml_import_dom($doc->importNode($reader->expand(), true));

            $word = strtolower((string)$node['word']);
            if ($word === $rhymeWith) {
				foreach ($node->pronunciations->pronunciation as $pronunciation) {
                    $rhymePronunciations[] = strtolower((string)$pronunciation);
                }
            }
			
            // Handle semanticallyClose
            if ($word === strtolower($semanticallyClose)) {
                if ($node->vectors) {
                    $targetVector = explode(' ', (string)$node->vectors);
					if ($reverseVectors) {
						$targetVector = array_map(function ($x) { return -$x; }, $targetVector);
					}
				}	
            }

			if ($vectorNode!=null && count($rhymePronunciations)>0) {
				break;
			}
						
        }
    }
}

$warnings = []; // To store warning messages
if ($rhymeWith && count($rhymePronunciations)==0) $warnings['rhymeWarning'] = "Prononciation pour le mot \"$rhymeWith\" introuvable.";
if ($semanticallyClose && !$targetVector) $warnings['semanticWarning'] = "Aucun vecteur trouvé pour le mot \"$semanticallyClose\".";

function cosineSimilarity(array $vecA, array $vecB): float {
    $dotProduct = 0;
    $magnitudeA = 0;
    $magnitudeB = 0;

    for ($i = 0; $i < count($vecA); $i++) {
        $dotProduct += $vecA[$i] * $vecB[$i];
        $magnitudeA += $vecA[$i] * $vecA[$i];
        $magnitudeB += $vecB[$i] * $vecB[$i];
    }

    $magnitudeA = sqrt($magnitudeA);
    $magnitudeB = sqrt($magnitudeB);

    // If magnitudes are zero, return zero to avoid division by zero
    if ($magnitudeA == 0 || $magnitudeB == 0) {
        return 0.0;
    }

    return $dotProduct / ($magnitudeA * $magnitudeB); // Cosine similarity
}

function euclidianDistance(array $vecA, array $vecB): float {
	$sum = 0;
	for ($i = 0; $i < count($vecA); $i++) {
		$sum += ($vecA[$i] - $vecB[$i]) * ($vecA[$i] - $vecB[$i]);
	}
	return sqrt($sum); // Euclidian distance
}

// Rewind the XMLReader to process the rest of the file
$reader->close();
$reader->open('words_database.xml');

$debugLog = "";

$seenWords = []; // Track already added words

while ($reader->read()) {
    if ($reader->nodeType == XMLReader::ELEMENT && $reader->name == 'word') {
        $doc = new DOMDocument();
        $node = simplexml_import_dom($doc->importNode($reader->expand(), true));

        $word = strtolower((string)$node['word']); // Normalize word to lowercase for comparison
		
        // Apply the "startsWith" filter
        if ($startsWith != '' && strpos($word, $startsWith) !== 0) {
            continue; // Skip words that don't start with the given prefix
        }
			
        // Filter by "natures"
        $nature = strtolower((string)$node['nature']);
        if (!empty($natures) && !in_array($nature, $natures)) {
            continue; // Skip words that don't match the selected natures
        }
		
        // Filter by "syllables"
        $syllables = (int)$node['syllables'];
        if (!empty($syllablesSelected) && !in_array($syllables, $syllablesSelected)) {
            continue; // Skip words that don't match the selected syllables
        }
				
		// Apply Genre filter
		if ($selectedGenre !== '') {
			if ($node['genre'] != $selectedGenre) {
				continue;
			}
		}
		
		// Apply Nombre filter
		if ($selectedNombre !== '') {
			if ($node['nombre'] != $selectedNombre) {
				continue;
			}
		}
		
        $frequency = (float)$node['freq']; // Get frequency value from XML
        $isInRange = false;// Check if frequency falls into any range
        foreach ($frequencyRanges as $range) {
			$rangeParts = explode('-', $range);
			if (count($rangeParts) !== 2) {
				continue; // Skip invalid range
			}
			$min = $rangeParts[0];
			$max = $rangeParts[1];
			$min = (float)$min;
			$max = ($max === 'Infinity') ? 1000000 : (float)$max;
			if ($frequency >= $min && $frequency <= $max) $isInRange = true;
        }
        if (!$isInRange) continue; // Skip if frequency is not in any range

	
		// Filter by "rhymeWith"
        if ($rhymeWith && $syllableSlider>0) {

            // If no pronunciations are found for the reference word, skip the filter
            if (count($rhymePronunciations)>0) {

				$wordPronunciations = [];
				$domNode = dom_import_simplexml($node); // Converts SimpleXMLElement to DOMElement
				$pronunciationsElement = $domNode->getElementsByTagName('pronunciations')->item(0);

				if ($pronunciationsElement) {
					$pronunciationElements = $pronunciationsElement->getElementsByTagName('pronunciation');
					foreach ($pronunciationElements as $p) {
						$wordPronunciations[] = strtolower($p->textContent); // Add pronunciation to the array
					}
				} else {
					continue;// skip if no pronunciation recorded
				}
				
				$matchesRhyme = false;
				foreach ($rhymePronunciations as $reference) {
					$referencePhonemes = array_slice(explode(' ', $reference), -$syllableSlider);
					foreach ($wordPronunciations as $pronunciation) {
						$wordPhonemes = array_slice(explode(' ', $pronunciation), -$syllableSlider);
						if ($referencePhonemes === $wordPhonemes) {
							$matchesRhyme = true;
							break;
						}
					}
					if ($matchesRhyme) {
						break;
					}
				}

				if (!$matchesRhyme) {
					continue; // Skip words that don't rhyme with the reference word
				}
			}

        }
		
		$score = 2.0; // Default score
		if ($semanticallyClose && $targetVector) {
			// Extract the vector of the candidate word
			$vectorElement = $node->vectors;				
			if ($vectorElement) {
				$vectorElement = explode(' ', $vectorElement);
				if (!$reverseVectors) $distance = (1 - cosineSimilarity($targetVector, $vectorElement))/2.0;// Compute cosine similarity
				else $distance = (1 - cosineSimilarity($targetVector, $vectorElement))/2.0;// Compute cosine similarity
				// else $distance = euclidianDistance($targetVector, $vectorElement);// if reversed vectors, compute euclidian distance
				$score = min($score,$distance);
			}
		}
		
        // Skip duplicates
        if (isset($seenWords[$word])) {
			$index = null; // Variable to store the found index
			foreach ($results as $key => $result) {
				if ($result['word'] === $word) {
					$index = $key; // Store the index if the word matches
					break; // Exit the loop once found
				}
			}
			if ($index !== null) {
				if ($score < $results[$index]['score']) {
					$results[$index]['score'] = $score; // Update to the better score
				}
				continue;
			}
        }

		$seenWords[$word] = true; // Mark word as seen
		
		// Limit results for performance
		if (count($results) < 1000) {
			$results[] = ['word' => $word, 'score' => $score];
		} else {
			// Find the lowest scored word in the results
			$maxIndex = null;
			$maxScore = 0;

			foreach ($results as $index => $result) {
				if ($result['score'] > $maxScore) {
					$maxScore = $result['score'];
					$maxIndex = $index;
				}
			}

			// If the new score is better than the lowest score, replace it
			if ($score < $maxScore) {
				$results[$maxIndex] = ['word' => $word, 'score' => $score];
			}
		}
	
    }
}

usort($results, function ($a, $b) {
    return $a['score'] <=> $b['score']; // Ascending order
});

$reader->close();

$results[] = ["debugLog" => $debugLog];

$response = [
    "results" => $results,
    "warnings" => $warnings
];

echo json_encode($response);

?>
