### Journal IA ###

| Outil | Ce que nous avons demandé | Réponse reçue | Vérification faite |Décision |

|---|---|---|---|---|

| ChatGPT | Comment entraîner le modèle et obtenir des prédictions? | 'fit' entraîne le modèle sur 'x_train' et 'y_train'; 'predict' produit des prédictions pour de nouvelles caractéristiques. | Nous avons vérifié ces appels dans 'src/entrainement.py' et exécuté les tests de prédiction. | Gardé. |

| ChatGPT | Comment utiliser le genre musical dans le modèle, puisqu'il s'agit d'une variable catégorielle? | 'OneHotEncoder' transforme les genres en colonnes numériques; le pipeline applique cet encodage avant la forêt aléatoire. | Nous avons entraîné le pipeline et vérifié qu’il produit des prédictions. | Gardé. |

| ChatGPT | Quel modèle trivial serait le plus approprié pour établir une référence : la classe majoritaire, la moyenne ou un autre? Pourquoi? | L’IA a recommandé la moyenne, car notre problème est une régression : la cible 'popularity' est numérique. La référence prédit la moyenne de 'y_train' pour chaque chanson de test. | Nous avons vérifié le calcul avec un exemple à la main et le test 'test_reference_calcule_mae'. | Gardé. |

| ChatGPT | Comment organiser la création des objets 'VariableNumerique', 'VariableCategorielle' et 'ValidateurVariables' entre l’interface et le module de validation? | L’IA a d’abord suggéré de créer ces objets directement dans l’interface. | En comparant nos fichiers, nous avons constaté qu’une de nous avait déjà écrit cette construction dans 'interface.py' et que l’autre avait aussi créé 'creer_validateur()' dans 'validation.py', ce qui créait une duplication. | Modifié : après avoir repéré cette duplication, nous avons plutôt gardé la création dans 'src/validation.py' et fait appeler 'creer_validateur()' par l’interface. |

| ChatGPT | Que signifie notre pourcentage de couverture et que pouvons-nous faire pour l'améliorer? | Il indique quelles lignes ont été exécutées par les tests; ce n’est pas une mesure directe de leur qualité. Un résultat de 40 % ne signifie pas nécessairement que les tests sont mauvais, puisque certaines parties, comme l’interface et le main.py, ne sont pas couvertes par les tests de logique. | Nous avons lancé Coverage et consulté les lignes manquantes dans son rapport. | Gardé : Nous avons décidé de ne pas ajouter de tests supplémentaires, puisque ce pourcentage est normal compte tenu de la structure de notre projet et qu’aucun seuil minimal n’est exigé. Nous avons plutôt conservé le résultat mesuré dans le README et expliqué les limites de la couverture.|

De façon plus générale, nous avons utilisé l'IA (notamment Claude et ChatGPT) pour nous expliquer certaines notions que nous n'avions pas vues dans le cadre du cours et que nous devions utiliser dans ce projet. Par exemple, elle nous a expliqué le fonctionnement d’un pipeline scikit-learn et de l’encodage avec OneHotEncoder, la différence entre fit et predict, ainsi que le principe d’une forêt aléatoire. Cela nous a permis de bien comprendre leur rôle et de savoir comment les implémenter de manière appropriée. 




