# Question 2: News Topic Clustering Practical Answer

## Dataset Understanding
- Records used in this run: 8,000.
- Key columns: `headline`, `short_description`, `category`, `authors`, `date`, and `link`.
- Average cleaned document length: 15.39 words.
- Vocabulary size after tokenization: 16,618 unique words.

The `headline` gives the strongest short topic signal, while `short_description` adds context. `category` is not used as a training label because the task is unsupervised, but it is useful for checking whether clusters make sense.

## First 10 Records
| headline | category | short_description |
| --- | --- | --- |
| Rick Snyder Says He's Releasing All His Emails About The Flint Water Crisis | POLITICS | It's not clear when the documents will be made public. |
| Mark Sanchez Pick Six: Bryan Scott Returns Interception For Touchdown In Jets-Bills (GIF) | SPORTS | For those who prefer their Jets fails as GIFs, here you go: On the Jets' second possession of the game -- the first resulted |
| Selena Gomez's AMAs Dress Is Very Sexy | STYLE | No surprises here. |
| Barbies May Limit Girls' Career Aspirations (STUDY) | PARENTING | The same held true regardless of whether the girls were playing with a Dr. Barbie or a Fashion Barbie, the researchers said |
| Check Out Jiff The Pomeranian's New Move | WEIRD NEWS | This is a showstopper. |
| April Superfoods: 5 In-Season Picks | WELLNESS | Still shaking off those winter doldrums and getting into the swing of spring? It's time to celebrate, with the season's best |
| I Visited 29 States In 90 Days For Just $3,600 | TRAVEL | This post appeared first on Centsai A couple of years ago, I drove from San Diego to Seattle to Washington, D.C. on a motorcycle |
| Ashton Kutcher And Mila Kunis' Matching Olympics Outfits Are So Damn Cute, They Deserve A Medal | ENTERTAINMENT | Dude, where's my gold? |
| 10 Great State Fairs For Deep-Fried Fun (PHOTOS) | TRAVEL | Don't limit yourself to just traditional fair fare: With vendors pushing the limits of what can be deep- fried -- from Coke to ribs -- sizzling new frontiers await. Join the festivities at these blue-ribbon-worthy state fairs. |
| Insensitive Washington Times Columnist Puts His Idiocy On Display | POLITICS | Charles Hurt attacks Jimmy Kimmel’s recent monologue opposing Donald Trump’s attempts to cut off funding for health care |

## Noise and Inconsistencies
- Mixed capitalization and punctuation create duplicate forms of the same word, such as `Trump`, `trump`, and `trump.`.
- Some descriptions are empty or very short, which can make document vectors weak and harder to cluster.
- Proper names, dates, numbers, and news-source style wording can dominate clusters unless normalized or filtered.

## Preprocessing Examples
- Original: Rick Snyder Says He's Releasing All His Emails About The Flint Water Crisis It's not clear when the documents will be made public.
  Cleaned: rick snyder releas email flint water crisi clear document public
- Original: Mark Sanchez Pick Six: Bryan Scott Returns Interception For Touchdown In Jets-Bills (GIF) For those who prefer their Jets fails as GIFs, here you go: On the Jets' second possession of the game -- the first resulted
  Cleaned: mark sanchez pick bryan scott return interception touchdown jet bill gif prefer jet fail gif jet second possession game result

Cleaning lowercases text, removes punctuation/symbols, tokenizes words, removes stopwords, and applies a simple lemmatization fallback. This improves clustering because documents about the same topic share more comparable tokens.

## Frequent Terms
| term | count |
| --- | --- |
| trump | 788 |
| photo | 589 |
| make | 450 |
| want | 439 |
| way | 428 |
| look | 366 |
| life | 362 |
| thing | 346 |
| know | 346 |
| need | 340 |

These terms summarize dominant news themes. Very general words are less useful, while named topics such as politics, entertainment, health, travel, or family-related terms help reveal clusters.

## Common Bigrams
| bigram | count |
| --- | --- |
| donald trump | 266 |
| white house | 74 |
| hillary clinton | 69 |
| health care | 51 |
| climate change | 51 |
| supreme court | 46 |
| unit state | 39 |
| social media | 37 |
| york city | 35 |
| super bowl | 33 |

## Common Trigrams
| trigram | count |
| --- | --- |
| want sure check | 24 |
| twitter facebook tumblr | 20 |
| facebook tumblr pinterest | 20 |
| sure check style | 17 |
| check style twitter | 17 |
| style twitter facebook | 17 |
| photo want sure | 12 |
| tumblr pinterest instagram | 12 |
| twitter facebook pinterest | 10 |
| black live matter | 10 |

N-grams preserve short phrases that single words lose. For example, a phrase can indicate a political institution, a celebrity-news theme, or a health topic more clearly than its separate words.

## Text Representation
- Bag-of-Words matrix shape: 8,000 documents x 5,000 terms.
- TF-IDF matrix shape: 8,000 documents x 5,000 terms.

Bag-of-Words stores raw counts, so repeated words receive high weight even if they are common. TF-IDF downweights words that appear in many documents and gives stronger weight to terms that distinguish one article from another. Both matrices are sparse because each article uses only a small part of the full vocabulary.

## Similar Document Pairs
| doc_a | doc_b | score | headline_a | headline_b | category_a | category_b |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | 120 | 0.42303337576555206 | Global Forgiveness Day: 5 Celebs Who Forgave Their Exes | Celebrity Exes: 6 Celeb Exes We'd Like To Cast In A Reality Show | DIVORCE | DIVORCE |
| 252 | 137 | 0.41004349414381336 | Republicans Have Always Been At War With The New York Times | The New York Times Just Provided A Massive Platform For Transphobia | POLITICS | QUEER VOICES |
| 192 | 280 | 0.39152430216039225 | Amber Rose's 19 Sexiest Social Media Snaps | Amber Rose Delivers Impassioned Speech About The First Time She Was Slut-Shamed | ENTERTAINMENT | WOMEN |
| 230 | 236 | 0.35760896407126863 | Laura Nekrasoe, Waitress, Swears By DIY Face Masks For Dry Skin | 7 DIY Revitalizing Beauty Tips for Hair and Skin | STYLE & BEAUTY | STYLE & BEAUTY |
| 245 | 110 | 0.3283581277874582 | Dad, Why Is that Man Wearing a Red Flower? | Hatchimals Are So Hard To Find That Parents Are Making Santa Explain | PARENTS | PARENTS |
| 204 | 220 | 0.3282322065267761 | Russian Trolls Didn't Need To Infiltrate The American Media, Because We Let Them In | Bury Lenin's Body: The Symbol of Communism Should No Longer Mock Humanity | MEDIA | WORLDPOST |
| 84 | 59 | 0.3100381745998725 | Death Toll From Italy Bridge Collapse Climbs To 37 | America's Most Dangerous Bridges (PHOTOS) | WORLD NEWS | TRAVEL |
| 112 | 74 | 0.2970475311691937 | Elin Nordegren: 10 Things She's Done Since The Tiger Woods Scandal | Joe Nocera: Another Week, Another Banking Scandal | DIVORCE | BUSINESS |
| 40 | 258 | 0.2821242433543881 | Georgia DA Could Bury Trump With His Own Words, Says Former Fed Prosecutor | Nirvana Jennette, Mom Forced Out Of Church For Breastfeeding, Aims To Change Georgia Law [UPDATED] | POLITICS | PARENTING |
| 288 | 144 | 0.27879754252016575 | Marcus Garvey's Message And Why A Pardon For Him Matters | Buckminster Fuller's Vision of a World That Works for Everyone | BLACK VOICES | ENTERTAINMENT |

Cosine similarity is high when two TF-IDF vectors point in a similar direction, usually because the documents share important words or phrases.

## Elbow and Cluster Selection
| k | inertia | silhouette |
| --- | --- | --- |
| 2 | 7935.959857706961 | 0.0013188818707594145 |
| 3 | 7908.009429764546 | 0.0020835001193386165 |
| 4 | 7894.054350098777 | 0.0023862630113127663 |
| 5 | 7882.962909733719 | 0.0027835693739150466 |
| 6 | 7872.354116203972 | 0.0028891202893537747 |
| 7 | 7862.986460864737 | 0.003236096339791634 |
| 8 | 7850.777745421792 | 0.0034852739982292134 |
| 9 | 7843.143338200328 | 0.0036813645843114715 |
| 10 | 7835.143378033716 | 0.0027100576723044184 |

Chosen k for the final TF-IDF K-Means model: 9. The plot is saved at `outputs/q2_elbow_tfidf.png` and includes a title, x-axis label, y-axis labels, legend, and grid.

## TF-IDF K-Means Cluster Interpretation
| cluster | size | topic_label | top_keywords | sample_headlines |
| --- | --- | --- | --- | --- |
| 0 | 5050 | Style, photos, and lifestyle media | make, look, know, life, want, women, thing, video, need, state | Rick Snyder Says He's Releasing All His Emails About The Flint Water Crisis \| Mark Sanchez Pick Six: Bryan Scott Returns Interception For Touchdown In Jets-Bills (GIF) \| Selena Gomez's AMAs Dress Is Very Sexy |
| 1 | 521 | Style, photos, and lifestyle media | photo, check, style, home, pinterest, twitter, facebook, look, fashion, wedd | Check Out Jiff The Pomeranian's New Move \| 10 Great State Fairs For Deep-Fried Fun (PHOTOS) \| Russian Cargo Ship Docks At International Space Station |
| 2 | 420 | Health and social issues | american, health, black, care, women, live, mental, man, study, country | Watch GOP Lawmakers Run Away When Asked If They Actually Read The Health Care Bill \| How Russia Often Benefits When Julian Assange Reveals the West’s Secrets \| These Are The 2 Key Ways People Get Hooked On Prescription Opioids |
| 3 | 100 | Politics and elections | clinton, hillary, trump, sander, campaign, bernie, donald, email, democratic, election | Hillary Clinton's Paid Speeches Were Totally At Odds With Her 2016 Platform \| Donald Trump Is 'Starting To Agree' Hillary Clinton Should Be Locked Up \| Hillary Clinton's Date Night Involved Applause And A Statement Necklace |
| 4 | 368 | Video and entertainment | world, watch, cup, thing, live, need, make, art, video, look | The Happiest Countries In The World (INFOGRAPHIC) \| Mediterranean Horrors: Fortress Europe's Vast Moat \| World's Dumbest Hijack Attempt? |
| 5 | 237 | Education, schools, and children | school, really, high, want, children, student, teacher, kid, think, college | We Tried It: SLT Yoga \| Justin Theroux Really, Really Hates Teva Sandals \| GOP Senate Candidate David Perdue Exaggerates His Father's Role Desegregating Georgia Schools |
| 6 | 271 | Parenting and family | kid, parent, mom, children, child, dad, need, want, teach, baby | Mom Of Transgender Teen Describes Her Experience As A Gift \| 10 Reasons Why It's OK To Spend New Year's Eve At Home, As Told In GIFs \| Getting a Grip: How to Take the Suckiness Out of January... and Get Your Groove Back |
| 7 | 552 | Politics and elections | trump, donald, president, republican, gop, white, house, campaign, democrat, russia | Insensitive Washington Times Columnist Puts His Idiocy On Display \| The Problem With Paternalizing Disabled People To Protest Donald Trump \| Khizr And Ghazala Khan Endorse Democrat In Virginia Governor's Race |
| 8 | 481 | Personal life and relationships | way, best, make, life, thing, work, love, better, don, want | April Superfoods: 5 In-Season Picks \| A Heart-to-Heart With the LGBTQ Community \| Gay One-Night Stands: Are They Keeping You From A Real Relationship? |

## Transformer Embedding Comparison
Embedding method used by this run: TruncatedSVD fallback used because SentenceTransformer was unavailable locally: ModuleNotFoundError: No module named 'sentence_transformers'

| cluster | size | topic_label | common_terms | sample_headlines |
| --- | --- | --- | --- | --- |
| 0 | 289 | Style, photos, and lifestyle media | look, photo, world, face, know, don, think, color | The Happiest Countries In The World (INFOGRAPHIC) \| Gills, Pills and Obesity Genes \| Table Talk: Local School Lunches |
| 1 | 4535 | World news and international affairs | state, world, american, right, health, help, school, live | Rick Snyder Says He's Releasing All His Emails About The Flint Water Crisis \| Mark Sanchez Pick Six: Bryan Scott Returns Interception For Touchdown In Jets-Bills (GIF) \| Selena Gomez's AMAs Dress Is Very Sexy |
| 2 | 1900 | Style, photos, and lifestyle media | photo, way, make, thing, life, know, want, best | April Superfoods: 5 In-Season Picks \| 10 Great State Fairs For Deep-Fried Fun (PHOTOS) \| Mom Of Transgender Teen Describes Her Experience As A Gift |
| 3 | 38 | Style, photos, and lifestyle media | check, twitter, facebook, pinterest, tumblr, photo, want, sure | Check Out Jiff The Pomeranian's New Move \| Michelle Obama At The 2008 Convention And Today: Has Her Look Changed? (PHOTOS) \| Taylor Schilling, 'The Lucky One' Star: Look Of The Day |
| 4 | 191 | Health and social issues | women, men, girl, work, american, want, world, busines | Youngest Assailant In Delhi Gang Rape Case Released From Prison \| Divorce Advice For Those About To Marry \| Obama Shares How Beyoncé Brings Sasha, Malia And The UConn Women's Basketball Team Together |
| 5 | 545 | Politics and elections | trump, donald, president, republican, gop, campaign, american, clinton | Insensitive Washington Times Columnist Puts His Idiocy On Display \| The Problem With Paternalizing Disabled People To Protest Donald Trump \| Khizr And Ghazala Khan Endorse Democrat In Virginia Governor's Race |
| 6 | 252 | Video and entertainment | video, watch, game, look, photo, home, man, way | Gloria Baume, 'Teen Vogue' Fashion Director Opens Her Closet (VIDEO) \| This Backyard Transforms From Plain To Party-Ready On A Budget (VIDEO) \| Paul Ryan Made A Video Hyping The GOP. But He Forgot Someone. |
| 7 | 61 | Health and social issues | sleep, night, study, hour, better, don, camp, health | Sleep Deprivation Linked With Buying More High-Calorie Foods While Grocery Shopping \| Ending the Life of a Pet \| The Good-Karma Guide to Holiday Travel |
| 8 | 189 | Politics and elections | house, white, trump, president, obama, republican, democrat, photo | Michelle Obama Reflects On ‘Scary’ Time At Princeton With Sweet Throwback Snap \| Watch GOP Lawmakers Run Away When Asked If They Actually Read The Health Care Bill \| Panda Express To Sub Brown Rice For White In Fried Rice Nationwide |

Transformer sentence embeddings usually improve clustering when they are available because they represent semantic meaning, not only exact shared words. For instance, documents about elections can be close even when one says `campaign` and another says `voters`. TF-IDF is easier to interpret and faster, but it is more sensitive to vocabulary mismatch.

## Evaluation Insight
The cluster keywords and sample headlines provide qualitative evaluation. A strong cluster has coherent keywords and documents that discuss the same topic. The original `category` column can be used only for after-the-fact validation because the clustering task itself is unsupervised. In this run, the very broad clusters show a known weakness of sparse TF-IDF K-Means on short news snippets: some documents share general vocabulary but not a single tight topic. The more specific clusters, such as politics/elections and parenting/school-related articles, are stronger because they contain distinctive repeated terms.