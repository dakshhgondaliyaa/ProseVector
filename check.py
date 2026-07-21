from datasets import load_dataset
ds = load_dataset('textminr/cmu-book-summaries', split='train')
titles = [item['title'] for item in ds]
targets = ['The Art of War', 'Meditations', 'The Origin of Species', 'Utilitarianism', 'Beyond Good and Evil', 'Thus Spoke Zarathustra', 'Communist Manifesto', 'Das Kapital', 'Capital', 'Silent Spring', 'The Selfish Gene', 'Guns, Germs, and Steel', "A People's History of the United States", "The Blank Slate", "Thinking, Fast and Slow", "A Brief History of Time", "The Structure of Scientific Revolutions", "The Interpretation of Dreams", "Orientalism"]
print([t for t in targets if t in titles])
