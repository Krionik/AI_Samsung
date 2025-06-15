plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    id("com.google.gms.google-services")
}

android {
    namespace = "com.example.facematch"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.facematch"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)

    // Supabase (Storage + Auth)
    implementation("io.github.jan-tennert.supabase:storage-kt:1.4.0")
    implementation("io.github.jan-tennert.supabase:postgrest-kt:1.4.0")
    implementation("io.github.jan-tennert.supabase:gotrue-kt:1.4.0") // Для авторизации (опционально)

    // Для работы с изображениями
    implementation("com.github.dhaval2404:imagepicker:2.1") // Выбор фото из галереи/камеры
    implementation("io.coil-kt:coil:2.4.0") // Загрузка и отображение изображений

    // Сетевые запросы (OkHttp)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation(libs.androidx.appcompat)
    implementation("io.ktor:ktor-client-okhttp:2.3.7")
    implementation("com.google.firebase:firebase-messaging:23.4.1")
    implementation(libs.firebase.messaging.ktx)

    implementation("androidx.localbroadcastmanager:localbroadcastmanager:1.1.0")
    implementation(libs.firebase.firestore.ktx)

    implementation("com.google.android.material:material:1.11.0")
    implementation("com.daimajia.androidanimations:library:2.4@aar")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}