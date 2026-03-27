import heatmapImg from '../assets/heatmap.png'

export default function HeatmapView() {
  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
      <div className="max-w-4xl mx-auto animate-fade-in">

        {/* Header */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-stone-800 dark:text-stone-100">
            Chicken Distribution Heatmap
          </h2>
          <p className="text-sm text-stone-500 dark:text-stone-400 mt-1">
            Latest heatmap from the CV pipeline
          </p>
        </div>

        {/* Image */}
        <div className="rounded-2xl border border-stone-200 dark:border-stone-700
                        bg-white dark:bg-stone-800 overflow-hidden">
          <div className="p-4">
            <img
              src={heatmapImg}
              alt="Chicken distribution heatmap"
              className="w-full object-contain max-h-[70vh] rounded-xl"
            />
          </div>
        </div>

      </div>
    </div>
  )
}
