import { useState } from 'react'
import bellaImg from '../assets/bella.png'
import bertImg from '../assets/bert.png'
import nuggetImg from '../assets/nugget.png'

const CHICKENS = [
  {
    name: 'Bella',
    image: bellaImg,
    breed: 'Barnevelder',
    age: '2 years',
    snack: 'Mealworms',
    eggsThisWeek: 5,
    bio: 'The undisputed boss of the coop. Bella inspects every corner before breakfast and judges you silently if you\'re late with the treats. Has a sixth sense for knowing exactly when you\'re about to clean and decides that\'s the perfect time to dust-bathe.',
  },
  {
    name: 'Bert',
    image: bertImg,
    breed: 'Brahma',
    age: '1.5 years',
    snack: 'Corn on the cob',
    eggsThisWeek: 0,
    bio: 'A gentle giant who genuinely believes he\'s a lap chicken. Follows you around the garden like a feathery shadow and occasionally tries to sit on the cat. Zero eggs, maximum personality. We don\'t talk about the time he got his head stuck in the fence.',
  },
  {
    name: 'Nugget',
    image: nuggetImg,
    breed: 'Silkie',
    age: '1 year',
    snack: 'Blueberries',
    eggsThisWeek: 3,
    bio: 'The escape artist. If there\'s a gap in the fence, Nugget has already found it. Fluffy, fast, and full of mischief. Once made it all the way to the neighbour\'s garden before anyone noticed. Looks innocent. Is not.',
  },
]

function ChickenCard({ chicken }) {
  const [flipped, setFlipped] = useState(false)

  return (
    <div
      className="w-full max-w-xs mx-auto cursor-pointer"
      style={{ perspective: '1000px' }}
      onClick={() => setFlipped(!flipped)}
    >
      <div
        className="relative w-full aspect-[3/4] transition-all duration-500"
        style={{
          transformStyle: 'preserve-3d',
          transform: flipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
        }}
      >
        {/* Front */}
        <div
          className="absolute inset-0 rounded-2xl border border-stone-200 dark:border-stone-700
                     bg-white dark:bg-stone-800 overflow-hidden shadow-sm
                     flex flex-col items-center justify-center p-6
                     hover:shadow-md hover:-translate-y-0.5 transition-shadow"
          style={{ backfaceVisibility: 'hidden' }}
        >
          <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-amber-200 dark:border-amber-600 mb-4">
            <img
              src={chicken.image}
              alt={chicken.name}
              className="w-full h-full object-cover"
            />
          </div>
          <h3 className="text-lg font-bold text-stone-800 dark:text-stone-100">
            {chicken.name}
          </h3>
          <p className="text-xs text-stone-400 dark:text-stone-500 mt-1">
            Tap to flip
          </p>
        </div>

        {/* Back */}
        <div
          className="absolute inset-0 rounded-2xl border border-stone-200 dark:border-stone-700
                     bg-white dark:bg-stone-800 overflow-hidden shadow-sm p-6
                     flex flex-col justify-between"
          style={{
            backfaceVisibility: 'hidden',
            transform: 'rotateY(180deg)',
          }}
        >
          <div>
            <h3 className="text-lg font-bold text-amber-600 dark:text-amber-400 mb-3">
              {chicken.name}
            </h3>
            <p className="text-sm text-stone-600 dark:text-stone-300 italic leading-relaxed">
              {chicken.bio}
            </p>
          </div>
          <div className="space-y-1.5 text-xs text-stone-500 dark:text-stone-400 mt-4 pt-4 border-t border-stone-100 dark:border-stone-700">
            <p>
              <span className="font-semibold text-stone-700 dark:text-stone-200">Breed:</span>{' '}
              {chicken.breed}
            </p>
            <p>
              <span className="font-semibold text-stone-700 dark:text-stone-200">Age:</span>{' '}
              {chicken.age}
            </p>
            <p>
              <span className="font-semibold text-stone-700 dark:text-stone-200">Fav snack:</span>{' '}
              {chicken.snack}
            </p>
            <p>
              <span className="font-semibold text-stone-700 dark:text-stone-200">Eggs this week:</span>{' '}
              {chicken.eggsThisWeek}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function AddChickenCard() {
  return (
    <div className="w-full max-w-xs mx-auto">
      <div className="w-full aspect-[3/4] rounded-2xl border-2 border-dashed border-stone-300 dark:border-stone-600
                      bg-stone-50 dark:bg-stone-800/50
                      flex flex-col items-center justify-center p-6 text-center
                      hover:border-amber-400 dark:hover:border-amber-500 transition-colors cursor-default">
        {/* Camera icon */}
        <svg
          className="w-12 h-12 text-stone-300 dark:text-stone-600 mb-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
          <circle cx="12" cy="13" r="4" />
        </svg>
        <h3 className="text-sm font-semibold text-stone-500 dark:text-stone-400 mb-1">
          Add a chicken
        </h3>
        <p className="text-xs text-stone-400 dark:text-stone-500 leading-relaxed">
          Upload a photo to add your next feathered friend to the flock
        </p>
      </div>
    </div>
  )
}

export default function MyChickens() {
  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto animate-fade-in">
        <h2 className="text-xl font-bold text-stone-800 dark:text-stone-100 mb-1">
          My Chickens
        </h2>
        <p className="text-sm text-stone-500 dark:text-stone-400 mb-6">
          Get to know your flock — tap a card to learn more
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {CHICKENS.map((c) => (
            <ChickenCard key={c.name} chicken={c} />
          ))}
          <AddChickenCard />
        </div>
      </div>
    </div>
  )
}
