import { useState } from 'react'
import { loginUser, registerUser } from '../api'
import logo from '../assets/chicken_logo_4x.png'

export default function LoginPage({ onLogin }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      if (mode === 'register') {
        await registerUser(username, password)
      }
      const data = await loginUser(username, password)
      localStorage.setItem('access_token', data.access_token)
      localStorage.setItem('username', data.username)
      onLogin(data.username)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-stone-50 dark:bg-stone-900 px-4">
      <div className="w-full max-w-sm bg-white dark:bg-stone-800 rounded-2xl shadow-lg p-8">

        <div className="flex flex-col items-center mb-6">
          <img src={logo} alt="ChickenCoopComfort" className="w-16 h-16 mb-3" />
          <h1 className="text-xl font-semibold text-stone-800 dark:text-stone-100">
            ChickenCoopComfort
          </h1>
          <p className="text-sm text-stone-500 dark:text-stone-400 mt-1">
            {mode === 'login' ? 'Sign in to your account' : 'Create a new account'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-stone-700 dark:text-stone-300 mb-1">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
              minLength={3}
              maxLength={50}
              className="w-full px-3 py-2 rounded-lg border border-stone-300 dark:border-stone-600
                         bg-white dark:bg-stone-700 text-stone-900 dark:text-stone-100
                         focus:outline-none focus:ring-2 focus:ring-amber-400"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-stone-700 dark:text-stone-300 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              minLength={8}
              maxLength={100}
              className="w-full px-3 py-2 rounded-lg border border-stone-300 dark:border-stone-600
                         bg-white dark:bg-stone-700 text-stone-900 dark:text-stone-100
                         focus:outline-none focus:ring-2 focus:ring-amber-400"
            />
          </div>

          {error && (
            <p className="text-sm text-red-500 dark:text-red-400">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 px-4 rounded-lg bg-amber-400 hover:bg-amber-500
                       text-stone-900 font-medium transition-colors
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
          </button>
        </form>

        <p className="text-center text-sm text-stone-500 dark:text-stone-400 mt-5">
          {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
          <button
            onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null) }}
            className="text-amber-500 hover:underline font-medium"
          >
            {mode === 'login' ? 'Register' : 'Sign in'}
          </button>
        </p>
      </div>
    </div>
  )
}
