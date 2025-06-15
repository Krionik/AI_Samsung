package com.example.facematch

import io.github.jan.supabase.SupabaseClient
import io.github.jan.supabase.createSupabaseClient
import io.github.jan.supabase.postgrest.Postgrest
import io.github.jan.supabase.storage.Storage

object Supabase {
    private const val SUPABASE_URL = "https://anzdioqdwsjwaoxirexi.supabase.co"
    private const val SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFuemRpb3Fkd3Nqd2FveGlyZXhpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc1MDE5NjYsImV4cCI6MjA2MzA3Nzk2Nn0.lWTu8J4g8Y8Wfv3sM4wAZdrcp7zOgWhgb3jTNSKKtWg"

    val client: SupabaseClient by lazy {
        createSupabaseClient(
            supabaseUrl = SUPABASE_URL,
            supabaseKey = SUPABASE_KEY
        ) {
            install(Storage)
            install(Postgrest)
        }
    }
}