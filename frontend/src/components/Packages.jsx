import { useState } from 'react'

function IconCheck({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  )
}

function IconX({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  )
}

function IconShoppingCart({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 2L3 6v14a2 2 0 002 2h14a2 2 0 002-2V6l-3-4z" />
      <path d="M3 6h18" />
      <path d="M16 10a4 4 0 01-8 0" />
    </svg>
  )
}
import standardImg from '../assets/pkg_standard.png'
import comfortImg from '../assets/pkg_comfort.png'
import predationImg from '../assets/pkg_predation.png'
import foodImg from '../assets/pkg_food.png'
import airqualityImg from '../assets/pkg_airquality.png'
import ultimateImg from '../assets/pkg_ultimate.png'

const PACKAGES = [
  {
    id: 'standard',
    name: 'Standard Package',
    tagline: "A beginner's kit for easier monitoring",
    price: 64.99,
    color: 'amber',
    badge: 'POPULAR',
    image: standardImg,
    description: "A beginner's approach to better welfare for your hens. By using a camera an eye is always kept on the coop. It informs you about possible diseases, broken eggs and general behavior of your chickens.",
    features: [
      'Temperature & humidity sensors',
      'Monitoring camera'
    ]
  },
  {
    id: 'comfort',
    name: 'Comfort Package',
    tagline: 'Standard with automation',
    price: 94.99,
    color: 'blue',
    image: comfortImg,
    description: 'An upgraded version of the standard package, which integrates automation for easier chicken keeping. The combination of temperature & humidity sensors with the ventilator will prevent the growth of mold in your coop.',
    features: [
      'Temperature & humidity sensors',
      'Monitoring camera',
      'Automated door',
      'Ventilator'
    ]
  },
  {
    id: 'predation',
    name: 'Predation Package',
    tagline: 'Protect against predators',
    price: 84.99,
    color: 'red',
    image: predationImg,
    description: 'This package ensures your chickens are protected against predators during day and night. Integrating a combination of safety measures & automation to secure a safe coop.',
    features: [
      'Monitoring camera',
      'Automated door',
      'Motion sensor light',
      'Eye imitation device',
      'Reflective tape'
    ]
  },
  {
    id: 'food',
    name: 'Food Package',
    tagline: 'Track feed & water intake',
    price: 124.99,
    color: 'green',
    image: foodImg,
    description: 'The focus of this package is tracking feed & water intake. It also uses a smart feeding system to take food scraps into account. This ensures that your chickens are well fed and always hydrated.',
    features: [
      'Automated feeder',
      'Drinker',
      'Ultrasonic sensors',
      'Monitoring camera'
    ]
  },
  {
    id: 'airquality',
    name: 'Air Quality Package',
    tagline: 'Clean air guaranteed',
    price: 64.99,
    color: 'cyan',
    image: airqualityImg,
    description: 'This package integrates several gas sensors that measure the concentration of important gasses for the chickens wellbeing. This way a healthy environment for your flock is guaranteed.',
    features: [
      'NH3 sensor (ammonia)',
      'H2S sensor (hydrogen sulfide)',
      'CO2 sensor',
      'Ventilator'
    ]
  },
  {
    id: 'ultimate',
    name: 'Ultimate Package',
    tagline: 'Full control of everything',
    price: 254.99,
    color: 'purple',
    badge: 'ALL-IN-ONE',
    image: ultimateImg,
    description: 'The Ultimate package is a combination of all previous packages. It ensures that your chickens are living their best lives, with full control from your phone.',
    features: [
      'Temperature & humidity sensors',
      'Monitoring camera',
      'Automated door',
      'Ventilator',
      'All 3 gas sensors',
      'Ultrasonic sensors',
      'Automated feeder',
      'Drinker',
      'Motion sensor light',
      'Eye imitation device',
      'Reflective tape'
    ]
  }
]

const COLOR_MAP = {
  amber: { bg: 'bg-amber-50 dark:bg-amber-500/10', border: 'border-amber-200 dark:border-amber-700', accent: 'bg-amber-400 dark:bg-amber-600', text: 'text-amber-700 dark:text-amber-400' },
  blue: { bg: 'bg-blue-50 dark:bg-blue-500/10', border: 'border-blue-200 dark:border-blue-700', accent: 'bg-blue-400 dark:bg-blue-600', text: 'text-blue-700 dark:text-blue-400' },
  red: { bg: 'bg-red-50 dark:bg-red-500/10', border: 'border-red-200 dark:border-red-700', accent: 'bg-red-400 dark:bg-red-600', text: 'text-red-700 dark:text-red-400' },
  green: { bg: 'bg-green-50 dark:bg-green-500/10', border: 'border-green-200 dark:border-green-700', accent: 'bg-green-400 dark:bg-green-600', text: 'text-green-700 dark:text-green-400' },
  cyan: { bg: 'bg-cyan-50 dark:bg-cyan-500/10', border: 'border-cyan-200 dark:border-cyan-700', accent: 'bg-cyan-400 dark:bg-cyan-600', text: 'text-cyan-700 dark:text-cyan-400' },
  purple: { bg: 'bg-purple-50 dark:bg-purple-500/10', border: 'border-purple-200 dark:border-purple-700', accent: 'bg-purple-400 dark:bg-purple-600', text: 'text-purple-700 dark:text-purple-400' }
}

function BasicSetCallout() {
  return (
    <div className="mb-10 relative overflow-hidden rounded-xl border border-stone-200 dark:border-amber-900/40 bg-white dark:bg-stone-900 shadow-sm">
      {/* Left accent bar */}
      <div className="absolute left-0 top-0 h-full w-1 bg-gradient-to-b from-amber-500 via-amber-400 to-amber-600" />

      <div className="flex items-start gap-4 pl-7 pr-6 py-5">
        {/* Icon */}
        <div className="shrink-0 flex items-center justify-center w-10 h-10 rounded-lg border border-amber-200 dark:border-amber-800/50 bg-amber-50 dark:bg-amber-950/40">
          <svg className="w-5 h-5 text-amber-600 dark:text-amber-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z" />
          </svg>
        </div>

        {/* Text */}
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-stone-800 dark:text-stone-100 mb-0.5">
            Basic Set Required
          </h3>
          <p className="text-sm text-stone-500 dark:text-stone-400 leading-relaxed">
            Every package requires the Basic Set (circuit, cabling &amp; casing) — the foundation all packages build on.
          </p>
        </div>

        {/* Price badge */}
        <div className="shrink-0 flex flex-col items-end justify-center ml-4">
          <div className="rounded-lg border border-amber-200 dark:border-amber-800/40 bg-amber-50 dark:bg-amber-950/30 px-4 py-2 text-center">
            <span className="block text-xl font-bold text-amber-700 dark:text-amber-400">€150</span>
            <span className="block text-[10px] font-medium text-stone-400 dark:text-stone-500 uppercase tracking-wide">one-time</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function PackageCard({ pkg, onSelect }) {
  const colors = COLOR_MAP[pkg.color]

  return (
    <div
      onClick={() => onSelect(pkg)}
      className={`cursor-pointer rounded-xl border ${colors.border} ${colors.bg} p-6
                   hover:shadow-lg hover:-translate-y-1 transition-all duration-200
                   flex flex-col h-full`}
    >
      {/* Image */}
      {pkg.image && (
        <div className="w-full h-48 mb-4 rounded-lg overflow-hidden bg-stone-100 dark:bg-stone-700 flex items-center justify-center">
          <img
            src={pkg.image}
            alt={pkg.name}
            className="w-full h-full object-contain p-2"
          />
        </div>
      )}

      {/* Top accent bar */}
      <div className={`w-12 h-1 rounded-full ${colors.accent} mb-4`}></div>

      {/* Badge */}
      {pkg.badge && (
        <div className={`inline-block w-fit mb-3 px-3 py-1 rounded-full text-xs font-bold ${colors.text}`}>
          {pkg.badge}
        </div>
      )}

      {/* Title & tagline */}
      <h3 className="text-lg font-bold text-stone-800 dark:text-stone-100 mb-1">
        {pkg.name}
      </h3>
      <p className="text-sm text-stone-600 dark:text-stone-400 mb-4">
        {pkg.tagline}
      </p>

      {/* Price */}
      <div className="mb-4">
        <p className={`text-2xl font-bold ${colors.text}`}>
          €{pkg.price.toFixed(2)}
        </p>
      </div>

      {/* Features preview */}
      <div className="mb-4 flex-1">
        <ul className="text-sm space-y-2">
          {pkg.features.slice(0, 3).map((feature, i) => (
            <li key={i} className="text-stone-600 dark:text-stone-400 flex items-center gap-2">
              <IconCheck className="w-4 h-4" /> {feature}
            </li>
          ))}
          {pkg.features.length > 3 && (
            <li className="text-stone-500 dark:text-stone-500 italic">+{pkg.features.length - 3} more</li>
          )}
        </ul>
      </div>

      {/* CTA hint */}
      <div className="text-sm font-medium text-stone-700 dark:text-stone-300">
        View details →
      </div>
    </div>
  )
}

function PackageDetailPanel({ pkg, isOpen, onClose }) {
  if (!pkg) return null

  const colors = COLOR_MAP[pkg.color]

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40 transition-opacity duration-200"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <div
        className={`fixed inset-y-0 right-0 w-full max-w-lg bg-white dark:bg-stone-800 z-50
                     flex flex-col overflow-hidden transition-transform duration-300
                     ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}
      >
        {/* Header with image */}
        <div className={`${colors.bg} border-b ${colors.border}`}>
          {pkg.image && (
            <div className="w-full h-64 overflow-hidden flex items-center justify-center">
              <img
                src={pkg.image}
                alt={pkg.name}
                className="w-full h-full object-contain p-4"
              />
            </div>
          )}
          <div className="p-6 flex items-start justify-between">
            <div>
              <h2 className="text-2xl font-bold text-stone-800 dark:text-stone-100 mb-1">
                {pkg.name}
              </h2>
              <p className={`text-lg font-bold ${colors.text}`}>
                €{pkg.price.toFixed(2)}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-stone-500 dark:text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 transition-colors"
            >
              <IconX className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Scrollable content */}
        <div className="overflow-y-auto flex-1 p-6">
          {/* Description */}
          <p className="text-stone-700 dark:text-stone-300 mb-6 leading-relaxed">
            {pkg.description}
          </p>

          {/* Features */}
          <div className="mb-8">
            <h3 className="font-bold text-stone-800 dark:text-stone-100 mb-4">What's included:</h3>
            <ul className="space-y-3">
              {pkg.features.map((feature, i) => (
                <li key={i} className="flex items-center gap-3 text-stone-700 dark:text-stone-300">
                  <Check className={`w-5 h-5 flex-shrink-0 ${colors.text}`} />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Footer CTA */}
        <div className="border-t border-stone-200 dark:border-stone-700 p-6">
          <button className={`w-full py-3 rounded-lg font-bold text-white flex items-center justify-center gap-2 ${colors.accent} hover:opacity-90 transition-opacity`}>
            <IconShoppingCart className="w-5 h-5" />
            Get Started
          </button>
          <p className="text-xs text-stone-500 dark:text-stone-400 text-center mt-3">
            Interested? Contact us or add to inquiry.
          </p>
        </div>
      </div>
    </>
  )
}

export default function Packages() {
  const [selectedPackage, setSelectedPackage] = useState(null)

  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-6xl mx-auto animate-fade-in">
        {/* Hero */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-stone-800 dark:text-stone-100 mb-4">
            Our Packages
          </h1>
          <p className="text-lg text-stone-600 dark:text-stone-400 max-w-2xl">
            Choose the package that best fits your chicken coop needs. Each package is modular and can be combined with others or stand alone.
          </p>
        </div>

        {/* Basic Set Callout */}
        <BasicSetCallout />

        {/* Package Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {PACKAGES.map(pkg => (
            <PackageCard
              key={pkg.id}
              pkg={pkg}
              onSelect={setSelectedPackage}
            />
          ))}
        </div>

        {/* Footer info */}
        <div className="text-center text-stone-500 dark:text-stone-400 text-sm">
          <p>All prices shown are base prices. Additional features and customization available upon request.</p>
        </div>
      </div>

      {/* Detail Panel */}
      <PackageDetailPanel
        pkg={selectedPackage}
        isOpen={!!selectedPackage}
        onClose={() => setSelectedPackage(null)}
      />
    </div>
  )
}
